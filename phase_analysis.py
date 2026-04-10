import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -----------------------------
# CONFIG
# -----------------------------
BIN_FILE = "/home/ck/Desktop/flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_TO_SHOW = 20.0
RANGE_BIN_LIMIT = 512
DISPLAY_STEP_CPI = 1
PAUSE_TIME = 0.08
CPI_INDEX_FOR_DEBUG = 0

# -----------------------------
# Helpers
# -----------------------------
def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            (buf[offset + 3]))

def read_u16_be(buf, offset):
    return ((buf[offset] << 8) |
            (buf[offset + 1]))

# -----------------------------
# Read file once
# -----------------------------
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File too small")

info = file_bytes[:INFO_SECTOR_SIZE]
raw_data = file_bytes[INFO_SECTOR_SIZE:]

# -----------------------------
# Decode info sector
# -----------------------------
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
ADC_SELECT         = info[idx]; idx += 1
USB_DATA_TYPE      = info[idx]; idx += 1
ADC_BITS           = info[idx]; idx += 1

CHIRP_END_TIMER_US      = read_u32_be(info, idx); idx += 4
CPI_END_TIMER_US        = read_u32_be(info, idx); idx += 4
CARD_WRITE_END_TIMER_US = read_u32_be(info, idx); idx += 4

CHIRPS_PER_CPI          = read_u16_be(info, idx); idx += 2
CPI_COUNTER             = read_u32_be(info, idx); idx += 4

# -----------------------------
# Derived values
# -----------------------------
FS = FS_KHZ * 1000
PRF = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6
fc = SWEEP_START + SWEEP_BW / 2.0
lam = 3e8 / fc

print("----- SYSTEM -----")
print(f"FS                : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"PRF               : {PRF:.2f} Hz")
print(f"HZ_PER_M          : {HZ_PER_M}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")

# -----------------------------
# Load ADC data
# -----------------------------
data = np.frombuffer(raw_data, dtype="<u2")

num_chirps = len(data) // SAMPLES_PER_CHIRP
data = data[:num_chirps * SAMPLES_PER_CHIRP]
chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

# center ADC
chirps = chirps.astype(np.float32) - 32768.0

num_cpis = num_chirps // CHIRPS_PER_CPI
if num_cpis == 0:
    raise ValueError("Not enough chirps for one CPI")

chirps = chirps[:num_cpis * CHIRPS_PER_CPI]
cpis = chirps.reshape(num_cpis, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)

print("\n----- DATA -----")
print("Num chirps :", num_chirps)
print("Num CPIs   :", num_cpis)

# -----------------------------
# FFT preparation
# -----------------------------
w_range = np.hanning(SAMPLES_PER_CHIRP).astype(np.float32)
w_dopp = np.hanning(CHIRPS_PER_CPI).astype(np.float32)

freq_hz_full = np.fft.fftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
freq_hz_full = freq_hz_full[:RANGE_BIN_LIMIT]
range_m_full = freq_hz_full / HZ_PER_M

valid_range_mask = (range_m_full >= 0) & (range_m_full <= MAX_RANGE_TO_SHOW)
range_m = range_m_full[valid_range_mask]

doppler_hz = np.fft.fftshift(np.fft.fftfreq(CHIRPS_PER_CPI, d=1.0 / PRF))
velocity_m_s = doppler_hz * lam / 2.0

# -----------------------------
# Debug CPI: select moving-target bin
# -----------------------------
if CPI_INDEX_FOR_DEBUG >= num_cpis:
    CPI_INDEX_FOR_DEBUG = 0

debug_cpi = cpis[CPI_INDEX_FOR_DEBUG].copy()

# analytic signal for better phase handling
debug_cpi = hilbert(debug_cpi, axis=1)

# remove per-chirp DC
debug_cpi = debug_cpi - np.mean(debug_cpi, axis=1, keepdims=True)

# range FFT
debug_rfft = np.fft.fft(debug_cpi * w_range[None, :], axis=1)
debug_rfft = debug_rfft[:, :RANGE_BIN_LIMIT]
debug_rfft[:, :5] = 0
debug_rfft = debug_rfft[:, valid_range_mask]

# clutter removal across chirps
debug_rfft_mti = debug_rfft - np.mean(debug_rfft, axis=0, keepdims=True)

# metrics
avg_mag = 20 * np.log10(np.mean(np.abs(debug_rfft), axis=0) + 1e-12)
moving_metric = 20 * np.log10(np.std(np.abs(debug_rfft_mti), axis=0) + 1e-12)

avg_mag -= np.max(avg_mag)
moving_metric -= np.max(moving_metric)

target_bin = int(np.argmax(moving_metric))
target_range_m = range_m[target_bin]

print("\n----- SELECTED BIN -----")
print(f"Debug CPI          : {CPI_INDEX_FOR_DEBUG}")
print(f"Target bin         : {target_bin}")
print(f"Target range       : {target_range_m:.2f} m")

# -----------------------------
# Phase analysis on selected bin
# -----------------------------
target_complex = debug_rfft[:, target_bin]
phase = np.angle(target_complex)
phase_unwrapped = np.unwrap(phase)
chirp_idx = np.arange(CHIRPS_PER_CPI)

fit = np.polyfit(chirp_idx, phase_unwrapped, 1)
phase_slope = fit[0]
phase_intercept = fit[1]
phase_fit = phase_slope * chirp_idx + phase_intercept
phase_residual = phase_unwrapped - phase_fit

fd_from_phase = (phase_slope / (2 * np.pi)) * PRF
vel_from_phase = fd_from_phase * lam / 2.0

print(f"Phase-slope fd     : {fd_from_phase:.2f} Hz")
print(f"Phase-slope vel    : {vel_from_phase:.2f} m/s")
print(f"Phase residual std : {np.std(phase_residual):.4f} rad")

# -----------------------------
# Doppler spectrum of selected bin
# -----------------------------
target_complex_mti = debug_rfft_mti[:, target_bin]
target_dopp = np.fft.fftshift(np.fft.fft(target_complex_mti * w_dopp))
target_dopp_mag = 20 * np.log10(np.abs(target_dopp) + 1e-12)
target_dopp_mag -= np.max(target_dopp_mag)

target_peak_idx = int(np.argmax(target_dopp_mag))
fd_fft_peak = doppler_hz[target_peak_idx]
vel_fft_peak = velocity_m_s[target_peak_idx]

print(f"FFT peak fd        : {fd_fft_peak:.2f} Hz")
print(f"FFT peak vel       : {vel_fft_peak:.2f} m/s")

# -----------------------------
# Plot debug figures
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(range_m, avg_mag, label="Average magnitude")
plt.plot(range_m, moving_metric, label="Moving-target metric")
plt.axvline(target_range_m, linestyle="--", label=f"Selected {target_range_m:.2f} m")
plt.xlabel("Range (m)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Range bin selection metrics - CPI {CPI_INDEX_FOR_DEBUG}")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 4))
plt.plot(chirp_idx, phase_unwrapped, label="Unwrapped phase")
plt.plot(chirp_idx, phase_fit, label="Linear fit")
plt.xlabel("Chirp index")
plt.ylabel("Phase (rad)")
plt.title(f"Phase vs chirp at selected range bin {target_bin} ({target_range_m:.2f} m)")
plt.grid(True)
plt.legend()

plt.figure(figsize=(10, 4))
plt.plot(chirp_idx, phase_residual)
plt.xlabel("Chirp index")
plt.ylabel("Residual phase (rad)")
plt.title("Phase residual after linear fit")
plt.grid(True)

plt.figure(figsize=(10, 4))
plt.plot(velocity_m_s, target_dopp_mag)
plt.axvline(vel_from_phase, linestyle="--", label="Phase-slope velocity")
plt.axvline(vel_fft_peak, linestyle=":", label="FFT peak velocity")
plt.xlabel("Velocity (m/s)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Doppler spectrum at selected range bin {target_bin}")
plt.grid(True)
plt.legend()

# -----------------------------
# Setup RDM video
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

dummy = np.zeros((CHIRPS_PER_CPI, len(range_m)), dtype=np.float32)

im = ax.imshow(
    dummy,
    aspect='auto',
    origin='lower',
    extent=[range_m[0], range_m[-1], velocity_m_s[0], velocity_m_s[-1]],
    vmin=-60,
    vmax=0
)

ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
title = ax.set_title("Range-Doppler Map")
fig.colorbar(im, ax=ax, label="dB")
fig.tight_layout()

# -----------------------------
# Main CPI-by-CPI video loop
# -----------------------------
for cpi_idx in range(0, num_cpis, DISPLAY_STEP_CPI):
    cpi = cpis[cpi_idx].copy()

    # analytic signal
    cpi = hilbert(cpi, axis=1)

    # remove per-chirp DC
    cpi = cpi - np.mean(cpi, axis=1, keepdims=True)

    # range FFT
    rfft = np.fft.fft(cpi * w_range[None, :], axis=1)
    rfft = rfft[:, :RANGE_BIN_LIMIT]
    rfft[:, :5] = 0
    rfft = rfft[:, valid_range_mask]

    # clutter removal across chirps
    rfft = rfft - np.mean(rfft, axis=0, keepdims=True)

    # Doppler FFT
    rdm = np.fft.fft(rfft * w_dopp[:, None], axis=0)
    rdm = np.fft.fftshift(rdm, axes=0)

    mag = 20 * np.log10(np.abs(rdm) + 1e-12)
    mag -= np.max(mag)

    im.set_data(mag)
    title.set_text(f"Range-Doppler Map - CPI {cpi_idx}")
    fig.canvas.draw_idle()
    plt.pause(PAUSE_TIME)

plt.ioff()
plt.show()