"""
GPU SAR processor for radar2v2 PRF/burst-mode .bin logs, with autofocus.

Parsing / CPI-cut / per-CPI-averaging / preprocessing sections are UNCHANGED
from your working script. Two NEW autofocus stages added on top, both purely
computational (no motion sensor needed):

  1. GLOBAL SPEED AUTOFOCUS (entropy search)
     Your entire azimuth grid (kx, delta_crange) is built from a single
     scalar `speed`. We search a range of candidate speeds, re-run Omega-K
     for each, and keep whichever produces the SHARPEST image (minimum
     Shannon entropy of |img|^2).

  2. PHASE GRADIENT AUTOFOCUS (PGA)
     Estimates and corrects RESIDUAL azimuth phase error (e.g. from the
     car not holding perfectly constant speed) directly from the image's
     own dominant scatterers, iteratively.

The azimuth FFT (`cfft`) does not depend on `speed` at all, so it is
computed ONCE and reused across every speed candidate -- the search is cheap.

Requires: cupy (matching your CUDA version), e.g.
  pip install cupy-cuda12x
"""

import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit(
        "cupy is not installed. Install the build matching your CUDA version, "
        "e.g.:  pip install cupy-cuda12x   (check `nvidia-smi` for your CUDA version)"
    ) from e

# ============================================================
# BIN FILE SETTINGS
# ============================================================
BIN_FILE = "fmcw2_bin_files/sar_log4.bin"

USE_SYNC_HEADERS = True
HEADER_SIZE = 4
SYNC0, SYNC1, SYNC2, SYNC3 = 0x1C1C, 0xC1C1, 0x9999, 0x00FF
INFO_SECTOR_SIZE = 512

# ============================================================
# SD-CARD RECORD CUT SETTINGS
# ============================================================
CUT_START_CPI = 0
CUT_END_CPI = 0
CUT_START_PERCENT = 0.00
CUT_END_PERCENT = 0.00

# ============================================================
# USER PARAMETERS
# ============================================================
speed = 0.8                 # initial guess, m/s -- autofocus will refine this

dynamic_range = 40
cross_range_padding = 4
ky_delta_spacing = 1.5
remove_leakage_bins = 5
remove_static_clutter = True
use_rvp = True

c = 299792458.0

# ============================================================
# AUTOFOCUS SETTINGS
# ============================================================
SPEED_SEARCH_ENABLE = True
SPEED_SEARCH_RANGE_FRAC = 0.6       # coarse search: +/- 60% around initial `speed`
SPEED_SEARCH_COARSE_STEPS = 25
SPEED_SEARCH_FINE_RANGE_FRAC = 0.1  # fine search: +/- 10% around coarse best
SPEED_SEARCH_FINE_STEPS = 15

PGA_ENABLE = True
PGA_ITERATIONS = 4
PGA_WINDOW_INIT_FRAC = 0.9          # fraction of azimuth extent used, shrinks each iter
PGA_WINDOW_SHRINK = 0.7
PGA_TOP_K_FRAC = 0.3                # fraction of range bins (by peak power) used for estimate

# ============================================================
# HELPERS
# ============================================================
def read_u32_be(buf, offset):
    return ((buf[offset] << 24) | (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) | buf[offset + 3])


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


def hilbert_rvp_gpu(x, fs, kr):
    y = cp.fft.fft(x, axis=-1)
    y[:, : y.shape[1] // 2 + 1] = 0
    f = cp.linspace(-fs / 2, fs / 2, y.shape[1])
    y *= cp.exp(-1j * cp.pi * f ** 2 / kr)
    return cp.fft.ifft(y, axis=-1)


def image_entropy(img):
    """Shannon entropy of normalized intensity. Lower = sharper/more focused."""
    p = cp.abs(img) ** 2
    total = cp.sum(p)
    if float(total) <= 0:
        return float("inf")
    p = p / total
    p_safe = cp.where(p > 1e-20, p, 1e-20)
    ent = -cp.sum(p * cp.log(p_safe))
    return float(ent)


def compute_ky_bin_count(kr, cpi_period, speed_val, ky_delta_spacing, n_az):
    """
    Ky-bin count for a nominal speed, used to FIX the array shape across every
    candidate in the speed search (see note in form_image_for_speed below).
    """
    delta_crange = speed_val * cpi_period
    kx = cp.linspace(-cp.pi / delta_crange, cp.pi / delta_crange, n_az)
    kr_delta = kr[1] - kr[0]
    kx_max = cp.max(cp.abs(kx))
    ky0 = cp.sqrt(cp.maximum(kr[0] ** 2 - kx_max ** 2, 1e-12))
    ky_delta = ky_delta_spacing * kr_delta
    n_bins = int(cp.arange(float(ky0), float(kr[-1]), float(ky_delta)).shape[0])
    return n_bins


def form_image_for_speed(cfft, kr, speed_val, cpi_period, ky_delta_spacing, n_ky_bins):
    """
    Runs Stolt interpolation + 2D IFFT for a given candidate speed.
    Returns the complex image (post ifft2+fftshift, BEFORE near/far reorder)
    and the resulting delta_crange.

    n_ky_bins is FIXED across every call (computed once, outside the speed
    search, from a nominal speed) rather than recomputed per-candidate via
    arange(). ky0 shifts slightly with every candidate speed, so a
    variable-step arange() gives a different array length each call --
    that defeats CuPy's memory pool (different shapes can't reuse freed
    blocks) and is what caused the OutOfMemoryError during the search.
    Using a fixed bin COUNT (via linspace) keeps every candidate's st/img
    arrays identically shaped, so the pool actually reuses memory.
    """
    delta_crange = speed_val * cpi_period
    n_az = cfft.shape[0]

    kx = cp.linspace(-cp.pi / delta_crange, cp.pi / delta_crange, n_az)

    val = kr[None, :] ** 2 - kx[:, None] ** 2
    valid = val > 0
    ky_full = cp.sqrt(cp.where(valid, val, 0.0))

    kx_max = cp.max(cp.abs(kx))
    ky0 = cp.sqrt(cp.maximum(kr[0] ** 2 - kx_max ** 2, 1e-12))
    ky_even = cp.linspace(float(ky0), float(kr[-1]), n_ky_bins)

    st = cp.zeros((n_az, n_ky_bins), dtype=cp.complex128)

    for i in range(n_az):
        v = valid[i]
        if int(cp.count_nonzero(v)) < 4:
            continue
        ky_row = ky_full[i][v]
        row = cfft[i][v]
        real_i = cp.interp(ky_even, ky_row, row.real, left=0.0, right=0.0)
        imag_i = cp.interp(ky_even, ky_row, row.imag, left=0.0, right=0.0)
        st[i, :] = real_i + 1j * imag_i

    del val, valid, ky_full, kx

    wx = cp.hanning(st.shape[0])
    wy = cp.hanning(st.shape[1])
    st = st * cp.sqrt(cp.outer(wx, wy))

    img = cp.fft.ifft2(st)
    img = cp.fft.fftshift(img, axes=0)
    del st
    return img, delta_crange


def pga_autofocus(img, n_iter=4, window_frac_init=0.9, window_shrink=0.7, top_k_frac=0.3):
    """
    Image-domain Phase Gradient Autofocus.
    Operates directly on the complex azimuth-compressed image. Each iteration:
      1. circularly center each range column's dominant scatterer
      2. build a windowed composite signal from the strongest range columns
      3. estimate azimuth phase error from that composite signal's spectrum
      4. remove linear trend (pure position shift, doesn't affect focus)
      5. apply the correction to every range column, shrink window, repeat
    """
    n_az, n_rg = img.shape
    corrected = img.copy()
    window_frac = window_frac_init

    print(f"PGA: starting entropy = {image_entropy(corrected):.4f}")

    for it in range(n_iter):
        mag = cp.abs(corrected)
        peak_idx = cp.argmax(mag, axis=0)                     # [n_rg]
        shift = (n_az // 2) - peak_idx

        idx = (cp.arange(n_az)[:, None] - shift[None, :]) % n_az
        shifted = cp.take_along_axis(corrected, idx, axis=0)

        peak_power = mag[peak_idx, cp.arange(n_rg)] ** 2
        k = max(8, int(top_k_frac * n_rg))
        top_idx = cp.argsort(peak_power)[-k:]

        win_len = max(8, int(window_frac * n_az))
        win = cp.zeros(n_az)
        start = n_az // 2 - win_len // 2
        win[start:start + win_len] = cp.hanning(win_len)

        g = cp.sum(shifted[:, top_idx] * win[:, None], axis=1)

        G = cp.fft.fft(g)
        phi = cp.unwrap(cp.angle(G))

        # remove linear trend -- that's just an azimuth position shift,
        # not a focus error, and shouldn't be "corrected" away
        kx_idx = cp.arange(n_az, dtype=cp.float64)
        A = cp.vstack([kx_idx, cp.ones(n_az)]).T
        coef, *_ = cp.linalg.lstsq(A, phi, rcond=None)
        phi_lin = A @ coef
        phi_corr = phi - phi_lin

        Icol_fft = cp.fft.fft(corrected, axis=0)
        Icol_fft *= cp.exp(-1j * phi_corr)[:, None]
        corrected = cp.fft.ifft(Icol_fft, axis=0)

        ent = image_entropy(corrected)
        print(f"PGA iter {it+1}/{n_iter}: residual phase std={float(cp.std(phi_corr)):.4f} rad, "
              f"entropy={ent:.4f}")

        window_frac *= window_shrink

    return corrected


# ============================================================
# READ BIN HEADER
# ============================================================
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File is smaller than 512-byte info sector")

info = file_bytes[:INFO_SECTOR_SIZE]

idx = 0

RECORD_COUNTER     = read_u32_be(info, idx); idx += 4
RECORD_TIME        = read_u32_be(info, idx); idx += 4
SWEEP_TIME_US      = read_u32_be(info, idx); idx += 4
SWEEP_GAP_US       = read_u32_be(info, idx); idx += 4
SWEEP_START_SCALED = read_u32_be(info, idx); idx += 4
SWEEP_BW_SCALED    = read_u32_be(info, idx); idx += 4
FS_KHZ             = read_u32_be(info, idx); idx += 4
SAMPLES_PER_CHIRP  = read_u32_be(info, idx); idx += 4

TX_MODE            = info[idx]; idx += 1
TX_POWER_DBM       = info[idx]; idx += 1
TX_POWER_DBM_VOLT  = info[idx]; idx += 1

HZ_PER_M           = read_u32_be(info, idx); idx += 4

DATA_LOG           = info[idx]; idx += 1
USB_DATA_TYPE      = info[idx]; idx += 1
ADC_BITS           = info[idx]; idx += 1

CHIRP_END_TIMER_US      = read_u32_be(info, idx); idx += 4
CPI_END_TIMER_US        = read_u32_be(info, idx); idx += 4
CARD_WRITE_END_TIMER_US = read_u32_be(info, idx); idx += 4

CHIRPS_PER_CPI = read_u16_be(info, idx); idx += 2
CPI_COUNTER    = read_u32_be(info, idx); idx += 4


if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

num_chirps_expected = CPI_COUNTER * CHIRPS_PER_CPI

if num_chirps_expected <= 0:
    raise ValueError("num_chirps_expected is zero")


FS = FS_KHZ * 1000
SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6

CONFIGURED_PRF_HZ = 0.0
if (SWEEP_TIME_US + SWEEP_GAP_US) > 0:
    CONFIGURED_PRF_HZ = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

MEASURED_CHIRP_RATE_HZ = 0.0
if CHIRP_END_TIMER_US > 0:
    MEASURED_CHIRP_RATE_HZ = 1e6 / CHIRP_END_TIMER_US

CPI_RATE_HZ = 0.0
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    CPI_RATE_HZ = 1e6 / (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US)


BYTES_PER_SAMPLE = 2

if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

BYTES_PER_CHIRP = words_per_chirp * BYTES_PER_SAMPLE
BYTES_PER_CPI = CHIRPS_PER_CPI * BYTES_PER_CHIRP

CONFIGURED_DATA_RATE_MBPS = (BYTES_PER_CHIRP * CONFIGURED_PRF_HZ) / 1e6

CARD_WRITE_SPEED_MBPS = 0.0
if CARD_WRITE_END_TIMER_US > 0:
    CARD_WRITE_SPEED_MBPS = BYTES_PER_CPI / (CARD_WRITE_END_TIMER_US / 1e6) / 1e6


print("\n----- SYSTEM -----")
print(f"FS                  : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"HEADER_SIZE         : {HEADER_SIZE if USE_SYNC_HEADERS else 0} words")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"ADC_BITS            : {ADC_BITS}")
print(f"TX_POWER            : {TX_POWER_DBM} dBm")
print(f"TX_VOLT             : {TX_POWER_DBM_VOLT}")
print(f"SWEEP_START         : {SWEEP_START/1e6:.2f} MHz")
print(f"SWEEP_BW            : {SWEEP_BW/1e6:.2f} MHz")

print("\n----- TIMING -----")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")

print("\n----- CPI -----")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_RATE            : {CPI_RATE_HZ:.2f} Hz")
print(f"CPI_COUNTER         : {CPI_COUNTER}")
print(f"NUM_CHIRPS          : {num_chirps_expected}")

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"BYTES_PER_CPI       : {BYTES_PER_CPI}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")

# -----------------------------
# Validate
# -----------------------------
if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

expected_chirps = CPI_COUNTER * CHIRPS_PER_CPI

if expected_chirps <= 0:
    raise ValueError("expected_chirps is zero")


# -----------------------------
# Print info
# -----------------------------
print("\n----- INFO -----")
print(f"FS                : {FS_KHZ / 1000:.3f} MHz")
print(f"ADC_BITS          : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS   : {expected_chirps}")


# -----------------------------
# Read only current record
# -----------------------------
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_to_read = expected_chirps * words_per_chirp * 2
print(f"Bytes to read: {bytes_to_read}")

raw_data = file_bytes[
    INFO_SECTOR_SIZE :
    INFO_SECTOR_SIZE + bytes_to_read
]
print(f"Raw data length: {len(raw_data)}")

adc_u16 = np.frombuffer(raw_data, dtype="<u2")
print(f"Raw data u16 length: {len(adc_u16)}")

# -----------------------------
# Extract chirps with fixed stride
# -----------------------------
available_chirps = len(adc_u16) // words_per_chirp
available_chirps = min(available_chirps, expected_chirps)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps available")

unused_words = len(adc_u16) - available_chirps * words_per_chirp

adc_u16 = adc_u16[:available_chirps * words_per_chirp]
chirps_raw = adc_u16.reshape(available_chirps, words_per_chirp)
full_cpi_count = available_chirps // CHIRPS_PER_CPI

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC0) |
        (chirps_raw[:, 1] != SYNC1) |
        (chirps_raw[:, 2] != SYNC2) |
        (chirps_raw[:, 3] != SYNC3)
    )[0]

    chirps = chirps_raw[:, HEADER_SIZE:]

    print("\n----- SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"BAD HEADERS        : {len(bad_headers)}")
    print(f"UNUSED END WORDS   : {unused_words}")
    print(f"VALID CPI          : {len(chirps) // CHIRPS_PER_CPI}")

    if len(bad_headers) > 0:
        print("First bad header indices:", bad_headers[:20])

else:
    chirps = chirps_raw

    print("\n----- NO SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"UNUSED END WORDS   : {unused_words}")
    print(f"VALID CPI          : {len(chirps) // CHIRPS_PER_CPI}")

num_chirps = len(chirps)

if num_chirps == 0:
    raise RuntimeError("No valid chirps found")

ADC_MASK = (1 << ADC_BITS) - 1
chirps = chirps & ADC_MASK
chirps = chirps.astype(np.float32)
chirps = (chirps / (2 ** ADC_BITS)) * 3.3

del chirps_raw, adc_u16, raw_data, file_bytes

# ============================================================
# CUT CPIs
# ============================================================
usable_chirps = full_cpi_count * CHIRPS_PER_CPI
chirps = chirps[:usable_chirps]

chirps_cpi = chirps.reshape(full_cpi_count, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)
del chirps

start_cpi = CUT_START_CPI
end_cpi = full_cpi_count - CUT_END_CPI
if end_cpi <= start_cpi:
    raise RuntimeError("Invalid CUT_START_CPI / CUT_END_CPI")
chirps_cpi = chirps_cpi[start_cpi:end_cpi]

p_start = int(chirps_cpi.shape[0] * CUT_START_PERCENT)
p_end = chirps_cpi.shape[0] - int(chirps_cpi.shape[0] * CUT_END_PERCENT)
if p_end <= p_start:
    raise RuntimeError("Invalid CUT_START_PERCENT / CUT_END_PERCENT")
chirps_cpi = chirps_cpi[p_start:p_end]

num_cpi_used = chirps_cpi.shape[0]
print(f"Using {num_cpi_used} CPIs")

# ============================================================
# COHERENT PER-CPI AVERAGE  <<< the key fix >>>
# The chirps inside a CPI sit at ~sub-mm spacing -- essentially the same
# physical position for imaging purposes. Collapse each CPI to ONE
# slow-time row. This removes the fabricated-interpolation problem and the
# memory blowup: aperture size goes from (num_CPI * chirps_per_CPI) rows
# down to just num_CPI rows.
# ============================================================
data1 = chirps_cpi.mean(axis=1)  # -> [num_cpi_used, SAMPLES_PER_CHIRP]
del chirps_cpi

# ============================================================
# TRUE CPI SPACING -> cross-range sample spacing
# ============================================================
chirp_period = (SWEEP_TIME_US + SWEEP_GAP_US) * 1e-6

if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    cpi_period = (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) * 1e-6
else:
    cpi_period = (
        CHIRPS_PER_CPI * (SWEEP_TIME_US + SWEEP_GAP_US) + CARD_WRITE_END_TIMER_US
    ) * 1e-6

print(f"chirp_period = {chirp_period*1e6:.1f} us")
print(f"cpi_period   = {cpi_period*1e3:.3f} ms")

# ============================================================
# MOVE TO GPU, PREPROCESS (unchanged, speed-independent)
# ============================================================
t_gpu_start = time.time()
data = cp.asarray(data1, dtype=cp.float64)
del data1

fs = FS
tsweep = SWEEP_TIME
bw = SWEEP_BW
fc = SWEEP_START + bw / 2.0
sweep_samples = data.shape[1]
lam = c / fc

print(f"Carrier {fc/1e9:.3f} GHz")

data = data - cp.mean(data, axis=1, keepdims=True)

wf = cp.hanning(sweep_samples)
data = data * wf

if use_rvp:
    data = hilbert_rvp_gpu(data, fs, bw / tsweep)

if remove_static_clutter:
    data = data - cp.mean(data, axis=0, keepdims=True)

if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1) * data.shape[0])
    data = cp.pad(data, ((zpad // 2, zpad // 2), (0, 0)), mode="constant")

ws = cp.hanning(data.shape[0])[:, None]
data = data * ws

# ============================================================
# AZIMUTH FFT -- computed ONCE, does not depend on speed
# ============================================================
kr = cp.linspace(
    (4.0 * cp.pi / c) * (fc - bw / 2.0),
    (4.0 * cp.pi / c) * (fc + bw / 2.0),
    sweep_samples,
)

cfft = cp.fft.fftshift(cp.fft.fft(data, axis=0), axes=0)

if remove_leakage_bins > 0:
    cfft[:, :remove_leakage_bins] = 0

# ============================================================
# STAGE 1: GLOBAL SPEED AUTOFOCUS (entropy search)
# ============================================================
best_speed = speed

# Fixed ky-bin count for the WHOLE search (see form_image_for_speed docstring):
# keeps every candidate's array shapes identical so CuPy's memory pool can
# actually reuse blocks instead of fragmenting/growing without bound.
n_ky_bins = compute_ky_bin_count(kr, cpi_period, speed, ky_delta_spacing, cfft.shape[0])
print(f"Fixed ky-bin count for autofocus search: {n_ky_bins}")

mempool = cp.get_default_memory_pool()

if SPEED_SEARCH_ENABLE:
    print("\n----- SPEED AUTOFOCUS (coarse) -----")
    t_search = time.time()
    coarse_candidates = np.linspace(
        speed * (1 - SPEED_SEARCH_RANGE_FRAC),
        speed * (1 + SPEED_SEARCH_RANGE_FRAC),
        SPEED_SEARCH_COARSE_STEPS,
    )
    coarse_candidates = coarse_candidates[coarse_candidates > 0]

    best_entropy = np.inf
    for sc in coarse_candidates:
        img_c, _ = form_image_for_speed(cfft, kr, float(sc), cpi_period, ky_delta_spacing, n_ky_bins)
        e = image_entropy(img_c)
        del img_c
        if e < best_entropy:
            best_entropy = e
            best_speed = float(sc)
    mempool.free_all_blocks()

    print(f"Coarse best speed: {best_speed:.4f} m/s ({best_speed*3.6:.2f} km/h), "
          f"entropy={best_entropy:.4f}")

    print("----- SPEED AUTOFOCUS (fine) -----")
    fine_candidates = np.linspace(
        best_speed * (1 - SPEED_SEARCH_FINE_RANGE_FRAC),
        best_speed * (1 + SPEED_SEARCH_FINE_RANGE_FRAC),
        SPEED_SEARCH_FINE_STEPS,
    )
    fine_candidates = fine_candidates[fine_candidates > 0]

    for sc in fine_candidates:
        img_c, _ = form_image_for_speed(cfft, kr, float(sc), cpi_period, ky_delta_spacing, n_ky_bins)
        e = image_entropy(img_c)
        del img_c
        if e < best_entropy:
            best_entropy = e
            best_speed = float(sc)
    mempool.free_all_blocks()

    print(f"Refined best speed: {best_speed:.4f} m/s ({best_speed*3.6:.2f} km/h), "
          f"entropy={best_entropy:.4f}")
    print(f"Speed search took {time.time()-t_search:.2f}s")

speed = best_speed

# ============================================================
# FORM FINAL IMAGE AT BEST SPEED
# ============================================================
img, delta_crange = form_image_for_speed(cfft, kr, speed, cpi_period, ky_delta_spacing, n_ky_bins)
print(f"\nFinal speed = {speed:.4f} m/s, delta_crange = {delta_crange*1000:.2f} mm "
      f"({delta_crange/lam:.4f} lambda)")

# ============================================================
# STAGE 2: PHASE GRADIENT AUTOFOCUS (residual, non-constant-velocity errors)
# ============================================================
if PGA_ENABLE:
    print("\n----- PHASE GRADIENT AUTOFOCUS -----")
    t_pga = time.time()
    img = pga_autofocus(
        img,
        n_iter=PGA_ITERATIONS,
        window_frac_init=PGA_WINDOW_INIT_FRAC,
        window_shrink=PGA_WINDOW_SHRINK,
        top_k_frac=PGA_TOP_K_FRAC,
    )
    print(f"PGA took {time.time()-t_pga:.2f}s")

# ============================================================
# IMAGE FORMATION / DISPLAY
# ============================================================
mid = img.shape[0] // 2
img_combined = cp.vstack((img[mid:, :], img[:mid, :]))

range1 = c * (fs / 2.0) * tsweep / (2.0 * bw)
crange = delta_crange * img_combined.shape[0]

img_db = 20 * cp.log10(cp.abs(img_combined) + 1e-12)
img_db_cpu = cp.asnumpy(img_db)

print(f"\nGPU pipeline total time: {time.time()-t_gpu_start:.2f}s")

plt.figure(figsize=(9, 7))
plt.title(f"SAR Image (GPU, autofocused, speed={speed:.3f} m/s)")
plt.imshow(
    img_db_cpu,
    aspect="auto",
    interpolation="none",
    extent=[0, range1, -crange / 2.0, crange / 2.0],
    origin="lower",
)
m = np.max(img_db_cpu)
plt.clim(m - dynamic_range, m)
plt.colorbar(label="dB")
plt.xlabel("Range [m]")
plt.ylabel("Cross-range [m]")
plt.tight_layout()
plt.show()