import time
import gc
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit("Install CuPy first: pip install cupy-cuda12x") from e


# ============================================================
# FILE
# ============================================================
BIN_FILE = "fmcw2_bin_files/sar_log4.bin"

USE_SYNC_HEADERS = True
HEADER_SIZE = 4
SYNC0, SYNC1, SYNC2, SYNC3 = 0x1C1C, 0xC1C1, 0x9999, 0x00FF
INFO_SECTOR_SIZE = 512


# ============================================================
# CUT NUMBER OF CPI FROM START AND END
# ============================================================
CUT_START_CPI = 0
CUT_END_CPI = 0


# ============================================================
# SAR SETTINGS FOR YOUR CAR SETUP
# ============================================================
ENABLE_SPEED_SEARCH = True

SPEED_SEARCH_MIN = 0.4
SPEED_SEARCH_MAX = 1.2
SPEED_SEARCH_STEPS = 17

speed = 0.8

cross_range_padding = 4
dynamic_range = 45

remove_leakage_bins = 0
remove_static_clutter = True
use_rvp = True

ENABLE_AUTOFOCUS = True
PGA_ITERATIONS = 8

FIXED_KY_BINS = 4096

c = 299792458.0


# ============================================================
# HELPERS
# ============================================================
def clear_gpu():
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def read_u32_be(buf, offset):
    return (
        (buf[offset] << 24)
        | (buf[offset + 1] << 16)
        | (buf[offset + 2] << 8)
        | buf[offset + 3]
    )


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


def hilbert_rvp_gpu(x, fs, kr_slope):
    y = cp.fft.fft(x, axis=-1)
    y[:, : y.shape[1] // 2 + 1] = 0

    f = cp.linspace(-fs / 2, fs / 2, y.shape[1], dtype=cp.float32)
    rvp = cp.exp(-1j * cp.pi * f**2 / kr_slope).astype(cp.complex64)

    return cp.fft.ifft(y, axis=-1).astype(cp.complex64) * rvp


def focus_metric(img):
    mag = cp.abs(img)

    # remove borders from metric
    a0 = img.shape[0] // 10
    a1 = img.shape[0] - a0
    r0 = max(5, img.shape[1] // 40)
    r1 = img.shape[1] - img.shape[1] // 20

    mag = mag[a0:a1, r0:r1]

    # sharpness / contrast metric
    p = mag**2
    p = p / (cp.mean(p) + 1e-12)

    return float(cp.mean(p**2).get())


def pga_autofocus(img, iterations=8):
    img = img.astype(cp.complex64, copy=True)

    n_az, n_rg = img.shape

    power_rg = cp.max(cp.abs(img), axis=0)
    th = cp.percentile(power_rg, 92)
    rg = cp.where(power_rg > th)[0]

    if int(rg.size) < 8:
        return img

    center = n_az // 2
    half = int(n_az * 0.30)

    az0 = max(0, center - half)
    az1 = min(n_az, center + half)

    for _ in range(iterations):
        sub = img[az0:az1, rg]

        peak = cp.argmax(cp.abs(sub), axis=0)
        aligned = cp.zeros_like(sub)

        ref = sub.shape[0] // 2

        for k in range(sub.shape[1]):
            shift = ref - int(peak[k].get())
            aligned[:, k] = cp.roll(sub[:, k], shift)

        spec = cp.fft.fftshift(cp.fft.fft(aligned, axis=0), axes=0)

        d = cp.conj(spec[:-1, :]) * spec[1:, :]
        grad = cp.angle(cp.sum(d, axis=1))

        phase = cp.cumsum(cp.concatenate([cp.zeros(1, dtype=grad.dtype), grad]))
        phase = cp.unwrap(phase)
        phase -= cp.mean(phase)

        full_phase = cp.interp(
            cp.arange(n_az, dtype=cp.float32),
            cp.linspace(az0, az1 - 1, phase.size, dtype=cp.float32),
            phase.astype(cp.float32),
        )

        img *= cp.exp(-1j * full_phase).astype(cp.complex64)[:, None]

    return img


def make_ky_grid_for_speed_range(kr, speed_min, cpi_period):
    """
    Fixed common ky grid.
    This removes the fake bias where different speeds gave different image sizes.
    """
    delta_min = speed_min * cpi_period
    kx_max = np.pi / delta_min

    kr0 = float(kr[0])
    kr1 = float(kr[-1])

    if kx_max >= kr0:
        ky_start = kr0 * 0.05
    else:
        ky_start = np.sqrt(kr0**2 - kx_max**2)

    ky_stop = kr1

    return cp.linspace(ky_start, ky_stop, FIXED_KY_BINS, dtype=cp.float32)


def form_image(data_cpu, delta_crange, fs, tsweep, bw, fc, ky_even, do_autofocus=True):
    data = cp.asarray(data_cpu, dtype=cp.float32)

    n_az0, n_samp = data.shape

    # fast-time DC
    data -= cp.mean(data, axis=1, keepdims=True)

    # range window
    data *= cp.hanning(n_samp).astype(cp.float32)[None, :]

    if use_rvp:
        data = hilbert_rvp_gpu(data, fs, bw / tsweep)

    if remove_static_clutter:
        data -= cp.mean(data, axis=0, keepdims=True)

    # azimuth padding
    if cross_range_padding > 1:
        zpad = int((cross_range_padding - 1) * data.shape[0])
        data = cp.pad(data, ((zpad // 2, zpad - zpad // 2), (0, 0)))

    # azimuth window
    data *= cp.hanning(data.shape[0]).astype(cp.float32)[:, None]

    n_az = data.shape[0]

    kx = cp.linspace(
        -cp.pi / delta_crange,
        cp.pi / delta_crange,
        n_az,
        dtype=cp.float32,
    )

    kr = cp.linspace(
        (4.0 * cp.pi / c) * (fc - bw / 2.0),
        (4.0 * cp.pi / c) * (fc + bw / 2.0),
        n_samp,
        dtype=cp.float32,
    )

    cfft = cp.fft.fftshift(cp.fft.fft(data, axis=0), axes=0).astype(cp.complex64)

    if remove_leakage_bins > 0:
        cfft[:, :remove_leakage_bins] = 0

    val = kr[None, :] ** 2 - kx[:, None] ** 2
    valid = val > 0
    ky_full = cp.sqrt(cp.where(valid, val, cp.float32(0.0))).astype(cp.float32)

    st = cp.zeros((n_az, ky_even.size), dtype=cp.complex64)

    for i in range(n_az):
        v = valid[i]

        if int(cp.count_nonzero(v).get()) < 4:
            continue

        ky_row = ky_full[i][v]
        row = cfft[i][v]

        re = cp.interp(ky_even, ky_row, row.real, left=0.0, right=0.0)
        im = cp.interp(ky_even, ky_row, row.imag, left=0.0, right=0.0)

        st[i, :] = (re + 1j * im).astype(cp.complex64)

    del data, cfft, val, valid, ky_full
    clear_gpu()

    # 2D window without giant outer temporary
    wx = cp.sqrt(cp.hanning(st.shape[0]).astype(cp.float32))
    wy = cp.sqrt(cp.hanning(st.shape[1]).astype(cp.float32))

    st *= wx[:, None]
    st *= wy[None, :]

    img = cp.fft.ifft2(st).astype(cp.complex64)
    img = cp.fft.fftshift(img, axes=0)

    del st
    clear_gpu()

    if do_autofocus:
        img = pga_autofocus(img, PGA_ITERATIONS)

    metric = focus_metric(img)

    mid = img.shape[0] // 2
    img = cp.vstack((img[mid:, :], img[:mid, :]))

    range_max = c * (fs / 2.0) * tsweep / (2.0 * bw)
    crange = delta_crange * img.shape[0]

    img_db = 20 * cp.log10(cp.abs(img) + 1e-12)
    img_db_cpu = cp.asnumpy(img_db)

    del img, img_db
    clear_gpu()

    return img_db_cpu, range_max, crange, metric


# ============================================================
# READ HEADER
# ============================================================
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

info = file_bytes[:INFO_SECTOR_SIZE]
idx = 0

RECORD_COUNTER_BIN = read_u32_be(info, idx); idx += 4
RECORD_TIME = read_u32_be(info, idx); idx += 4
SWEEP_TIME_US = read_u32_be(info, idx); idx += 4
SWEEP_DELAY_US = read_u32_be(info, idx); idx += 4
SWEEP_START_SCALED = read_u32_be(info, idx); idx += 4
SWEEP_BW_SCALED = read_u32_be(info, idx); idx += 4
FS_KHZ = read_u32_be(info, idx); idx += 4
NUMBER_OF_SAMPLES = read_u32_be(info, idx); idx += 4

TX_MODE = info[idx]; idx += 1
TX_POWER_DBM = info[idx]; idx += 1
TX_POWER_DBM_VOLTAGE = info[idx]; idx += 1

hz_per_m = read_u32_be(info, idx); idx += 4

DATA_LOG = info[idx]; idx += 1
USB_DATA_TYPE = info[idx]; idx += 1
ADC_RESOLUTION = info[idx]; idx += 1

CHIRP_END_TIMER_US = read_u32_be(info, idx); idx += 4
CPI_END_TIMER_US = read_u32_be(info, idx); idx += 4
CARD_WRITE_END_TIMER_US = read_u32_be(info, idx); idx += 4

CHIRPS_PER_CPI = read_u16_be(info, idx); idx += 2
CPI_COUNTER = read_u32_be(info, idx); idx += 4

RECORD_COUNTER_HEADER = CPI_COUNTER * CHIRPS_PER_CPI

SWEEP_TIME = SWEEP_TIME_US / 1e6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6
SAMPLING_FREQUENCY = FS_KHZ * 1000.0

print("----- HEADER -----")
print(f"CHIRPS_PER_CPI          : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER             : {CPI_COUNTER}")
print(f"SWEEP_TIME / DELAY      : {SWEEP_TIME_US} us / {SWEEP_DELAY_US} us")
print(f"CPI_END_TIMER_US        : {CPI_END_TIMER_US}")
print(f"CARD_WRITE_END_TIMER_US : {CARD_WRITE_END_TIMER_US}")
print(f"SAMPLING_FREQUENCY      : {SAMPLING_FREQUENCY}")
print(f"SAMPLES_PER_CHIRP       : {NUMBER_OF_SAMPLES}")
print(f"SWEEP_BW / START        : {SWEEP_BW/1e6:.2f} MHz / {SWEEP_START/1e6:.2f} MHz")
print(f"ADC_RESOLUTION          : {ADC_RESOLUTION}")


# ============================================================
# READ ADC DATA
# ============================================================
words_per_chirp = NUMBER_OF_SAMPLES + (HEADER_SIZE if USE_SYNC_HEADERS else 0)

raw_data = file_bytes[INFO_SECTOR_SIZE:]
data_u16 = np.frombuffer(raw_data, dtype="<u2")

available_chirps = len(data_u16) // words_per_chirp
available_chirps = min(available_chirps, RECORD_COUNTER_HEADER)

full_cpi_count = available_chirps // CHIRPS_PER_CPI
usable_chirps = full_cpi_count * CHIRPS_PER_CPI

print(f"available_chirps = {available_chirps}")
print(f"full_cpi_count   = {full_cpi_count}")

data_u16 = data_u16[: usable_chirps * words_per_chirp]
chirps_raw = data_u16.reshape(usable_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad = np.where(
        (chirps_raw[:, 0] != SYNC0)
        | (chirps_raw[:, 1] != SYNC1)
        | (chirps_raw[:, 2] != SYNC2)
        | (chirps_raw[:, 3] != SYNC3)
    )[0]

    print(f"bad headers: {len(bad)}")

    chirps = chirps_raw[:, HEADER_SIZE:]
else:
    chirps = chirps_raw

ADC_MASK = (1 << ADC_RESOLUTION) - 1
chirps = chirps & ADC_MASK
chirps = chirps.astype(np.float32)
chirps = (chirps / (2**ADC_RESOLUTION)) * 3.3

del chirps_raw, data_u16, raw_data, file_bytes


# ============================================================
# CPI AVERAGE
# ============================================================
chirps_cpi = chirps.reshape(full_cpi_count, CHIRPS_PER_CPI, NUMBER_OF_SAMPLES)
del chirps

chirps_cpi = chirps_cpi[CUT_START_CPI : full_cpi_count - CUT_END_CPI]

print(f"Using {chirps_cpi.shape[0]} CPIs")

data1 = chirps_cpi.mean(axis=1).astype(np.float32)
del chirps_cpi


# ============================================================
# APERTURE
# ============================================================
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    cpi_period = (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) * 1e-6
else:
    cpi_period = (
        CHIRPS_PER_CPI * (SWEEP_TIME_US + SWEEP_DELAY_US)
        + CARD_WRITE_END_TIMER_US
    ) * 1e-6

fs = SAMPLING_FREQUENCY
tsweep = SWEEP_TIME
bw = SWEEP_BW
fc = SWEEP_START + bw / 2.0
lam = c / fc

print("----- APERTURE -----")
print(f"cpi_period : {cpi_period*1e3:.3f} ms")
print(f"carrier    : {fc/1e9:.3f} GHz")
print(f"lambda     : {lam*1000:.2f} mm")


# fixed common ky grid for fair speed search
kr_cpu = np.linspace(
    (4.0 * np.pi / c) * (fc - bw / 2.0),
    (4.0 * np.pi / c) * (fc + bw / 2.0),
    NUMBER_OF_SAMPLES,
    dtype=np.float32,
)

kr_gpu = cp.asarray(kr_cpu)
ky_even = make_ky_grid_for_speed_range(kr_gpu, SPEED_SEARCH_MIN, cpi_period)


# ============================================================
# SPEED SEARCH
# ============================================================
if ENABLE_SPEED_SEARCH:
    print("----- SPEED SEARCH -----")

    speeds = np.linspace(SPEED_SEARCH_MIN, SPEED_SEARCH_MAX, SPEED_SEARCH_STEPS)

    metrics = []
    best_speed = speeds[0]
    best_metric = -1e99

    for s in speeds:
        t0 = time.time()

        delta = s * cpi_period

        try:
            _, _, _, metric = form_image(
                data1,
                delta,
                fs,
                tsweep,
                bw,
                fc,
                ky_even,
                do_autofocus=True,
            )

        except Exception as e:
            print(f"speed={s:.3f} failed: {e}")
            metric = -1e99

        clear_gpu()

        metrics.append(metric)

        print(
            f"speed={s:.3f} m/s | "
            f"delta={delta*1000:.2f} mm/CPI | "
            f"metric={metric:.3f} | "
            f"time={time.time()-t0:.2f}s"
        )

        if metric > best_metric:
            best_metric = metric
            best_speed = s

    speed = best_speed

    print(f"BEST SPEED = {speed:.3f} m/s")

    plt.figure(figsize=(7, 4))
    plt.plot(speeds, metrics, marker="o")
    plt.grid(True)
    plt.xlabel("Speed [m/s]")
    plt.ylabel("Focus metric")
    plt.title("Fair Speed Search, Fixed ky Grid")
    plt.tight_layout()
    plt.show()


# ============================================================
# FINAL IMAGE
# ============================================================
delta_crange = speed * cpi_period

print("----- FINAL IMAGE -----")
print(f"speed        : {speed:.3f} m/s")
print(f"delta_crange : {delta_crange*1000:.2f} mm/CPI")
print(f"delta/lambda : {delta_crange/lam:.3f}")

t0 = time.time()

img_db, range_max, crange, metric = form_image(
    data1,
    delta_crange,
    fs,
    tsweep,
    bw,
    fc,
    ky_even,
    do_autofocus=True,
)

print(f"final metric : {metric:.3f}")
print(f"total time   : {time.time()-t0:.2f}s")


# ============================================================
# PLOT SAR IMAGE
# ============================================================
plt.figure(figsize=(10, 7))
plt.title(f"SAR Image | speed={speed:.3f} m/s | PGA autofocus")
plt.imshow(
    img_db,
    aspect="auto",
    interpolation="none",
    extent=[0, range_max, -crange / 2, crange / 2],
    origin="lower",
)

m = np.max(img_db)
plt.clim(m - dynamic_range, m)

plt.colorbar(label="dB")
plt.xlabel("Range [m]")
plt.ylabel("Cross-range [m]")
plt.tight_layout()
plt.show()