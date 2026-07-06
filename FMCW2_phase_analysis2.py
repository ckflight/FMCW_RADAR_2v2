import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================
OPERATING_SYSTEM = 1

USE_SYNC_HEADERS = True
HEADER_SIZE = 4

SYNC0 = 0x1C1C
SYNC1 = 0xC1C1
SYNC2 = 0x9999
SYNC3 = 0x00FF

if OPERATING_SYSTEM == 1:
    BIN_FILE = "Radar_Records/data_record.bin"
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    #BIN_FILE = "fmcw2_bin_files/hwfir_terrace.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512

AUTO_PICK_BIN = True
FORCED_BIN = 54

IGNORE_FIRST_BINS = 5
MAX_RANGE_TO_SHOW = 100
AVG_CHIRPS_FOR_BIN_PICK = 512

REMOVE_FAST_TIME_DC = True
WINDOW_TYPE = "hann"   # "hann" or "rect"

# For quality metric
REMOVE_LINEAR_PHASE_TREND = True

# ============================================================
# HELPERS
# ============================================================
def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            buf[offset + 3])


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def detrend_linear(y):
    n = np.arange(len(y))
    p = np.polyfit(n, y, 1)
    fit = np.polyval(p, n)
    return y - fit, fit, p


# ============================================================
# READ FILE
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

FS = FS_KHZ * 1000.0
num_chirps_expected = CPI_COUNTER * CHIRPS_PER_CPI

if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise RuntimeError("SAMPLES_PER_CHIRP is zero")

if num_chirps_expected <= 0:
    raise RuntimeError("CPI_COUNTER or CHIRPS_PER_CPI is zero")

SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6

CONFIGURED_PRF_HZ = 0.0
if (SWEEP_TIME_US + SWEEP_GAP_US) > 0:
    CONFIGURED_PRF_HZ = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

MEASURED_CHIRP_RATE_HZ = 0.0
if CHIRP_END_TIMER_US > 0:
    MEASURED_CHIRP_RATE_HZ = 1e6 / CHIRP_END_TIMER_US

CPI_RATE_HZ = 0.0
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    CPI_RATE_HZ = 1e6 / (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US)

print("\n========== SYSTEM ==========")
print(f"FS                  : {FS/1e6:.3f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"ADC_BITS            : {ADC_BITS}")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER         : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS     : {num_chirps_expected}")


# ============================================================
# READ ADC DATA
# ============================================================
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_to_read = num_chirps_expected * words_per_chirp * 2

raw_data = file_bytes[
    INFO_SECTOR_SIZE:
    INFO_SECTOR_SIZE + bytes_to_read
]

data_u16 = np.frombuffer(raw_data, dtype="<u2")

available_chirps = len(data_u16) // words_per_chirp
available_chirps = min(available_chirps, num_chirps_expected)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps available")

unused_words = len(data_u16) - available_chirps * words_per_chirp

data_u16 = data_u16[:available_chirps * words_per_chirp]
chirps_raw = data_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC0) |
        (chirps_raw[:, 1] != SYNC1) |
        (chirps_raw[:, 2] != SYNC2) |
        (chirps_raw[:, 3] != SYNC3)
    )[0]

    chirps = chirps_raw[:, HEADER_SIZE:]

    print("\n========== SYNC ==========")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"BAD HEADERS        : {len(bad_headers)}")
    print(f"UNUSED END WORDS   : {unused_words}")

    if len(bad_headers) > 0:
        print("First bad header indices:", bad_headers[:30])
else:
    chirps = chirps_raw
    bad_headers = np.array([], dtype=int)

    print("\n========== NO SYNC ==========")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"UNUSED END WORDS   : {unused_words}")

num_chirps = len(chirps)

ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

chirps = chirps & ADC_MASK
chirps = chirps.astype(np.float32) - ADC_CENTER

num_cpis = num_chirps // CHIRPS_PER_CPI
num_chirps_used = num_cpis * CHIRPS_PER_CPI

if num_cpis == 0:
    raise RuntimeError("Not enough chirps for one full CPI")

chirps = chirps[:num_chirps_used]

print("\n========== DATA ==========")
print(f"Num chirps loaded : {num_chirps}")
print(f"Num chirps used   : {num_chirps_used}")
print(f"Num CPIs          : {num_cpis}")


# ============================================================
# RANGE FFT
# ============================================================
if WINDOW_TYPE == "hann":
    w = np.hanning(SAMPLES_PER_CHIRP).astype(np.float32)
else:
    w = np.ones(SAMPLES_PER_CHIRP, dtype=np.float32)

x = chirps.copy()

if REMOVE_FAST_TIME_DC:
    x = x - np.mean(x, axis=1, keepdims=True)

x = x * w[None, :]

range_fft = np.fft.rfft(x, axis=1)

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

num_bins = range_fft.shape[1]


# ============================================================
# PICK TARGET BIN
# ============================================================
num_avg = min(AVG_CHIRPS_FOR_BIN_PICK, range_fft.shape[0])
avg_mag = np.mean(np.abs(range_fft[:num_avg, :]), axis=0)

valid = np.where(range_m <= MAX_RANGE_TO_SHOW)[0]
valid = valid[valid >= IGNORE_FIRST_BINS]

if len(valid) == 0:
    raise RuntimeError("No valid range bins")

if AUTO_PICK_BIN:
    track_bin = valid[np.argmax(avg_mag[valid])]
else:
    track_bin = FORCED_BIN

track_range_m = range_m[track_bin]
track_freq_hz = freq_hz[track_bin]

# SNR-like bin metric
noise_bins = valid[valid != track_bin]
noise_floor = np.median(avg_mag[noise_bins]) + 1e-12
target_mag = avg_mag[track_bin]
target_snr_db = 20 * np.log10(target_mag / noise_floor)

print("\n========== TRACKED TARGET ==========")
print(f"track_bin          : {track_bin}")
print(f"track_freq         : {track_freq_hz/1e3:.2f} kHz")
print(f"track_range        : {track_range_m:.2f} m")
print(f"target/bin SNR     : {target_snr_db:.2f} dB")


# ============================================================
# PHASE PERFORMANCE MEASUREMENT
# ============================================================
tracked = range_fft[:, track_bin]

phase_wrapped = np.angle(tracked)
phase_unwrapped = np.unwrap(phase_wrapped)

amp = np.abs(tracked)
amp_db = 20 * np.log10(amp + 1e-12)

n = np.arange(len(phase_unwrapped))

if REMOVE_LINEAR_PHASE_TREND:
    phase_error, phase_fit, phase_poly = detrend_linear(phase_unwrapped)
    amp_error_db, amp_fit_db, amp_poly = detrend_linear(amp_db)
else:
    phase_error = phase_unwrapped - np.mean(phase_unwrapped)
    phase_fit = np.full_like(phase_unwrapped, np.mean(phase_unwrapped))
    amp_error_db = amp_db - np.mean(amp_db)
    amp_fit_db = np.full_like(amp_db, np.mean(amp_db))
    phase_poly = [0.0, np.mean(phase_unwrapped)]

phase_jitter_rad = rms(phase_error)
phase_jitter_deg = np.rad2deg(phase_jitter_rad)

phase_peak_to_peak_rad = np.max(phase_error) - np.min(phase_error)
phase_peak_to_peak_deg = np.rad2deg(phase_peak_to_peak_rad)

amp_jitter_db = rms(amp_error_db)
amp_peak_to_peak_db = np.max(amp_error_db) - np.min(amp_error_db)

# Phase step chirp-to-chirp
phase_step = np.angle(tracked[1:] * np.conj(tracked[:-1]))
phase_step_unwrapped = np.unwrap(phase_step)
phase_step_mean_rad = np.mean(phase_step)
phase_step_std_rad = np.std(phase_step)

# Convert linear phase slope to apparent Doppler
# phase slope per chirp = 2*pi*fd/PRF
phase_slope_rad_per_chirp = phase_poly[0]

if CONFIGURED_PRF_HZ > 0:
    apparent_fd_hz = phase_slope_rad_per_chirp * CONFIGURED_PRF_HZ / (2 * np.pi)
elif MEASURED_CHIRP_RATE_HZ > 0:
    apparent_fd_hz = phase_slope_rad_per_chirp * MEASURED_CHIRP_RATE_HZ / (2 * np.pi)
else:
    apparent_fd_hz = np.nan

print("\n========== PHASE PERFORMANCE ==========")
print(f"Phase jitter RMS        : {phase_jitter_rad:.6f} rad")
print(f"Phase jitter RMS        : {phase_jitter_deg:.3f} deg")
print(f"Phase error p-p         : {phase_peak_to_peak_rad:.6f} rad")
print(f"Phase error p-p         : {phase_peak_to_peak_deg:.3f} deg")
print(f"Linear phase slope      : {phase_slope_rad_per_chirp:.6e} rad/chirp")
print(f"Apparent Doppler        : {apparent_fd_hz:.3f} Hz")
print(f"Chirp phase-step mean   : {phase_step_mean_rad:.6f} rad")
print(f"Chirp phase-step std    : {phase_step_std_rad:.6f} rad")
print(f"Chirp phase-step std    : {np.rad2deg(phase_step_std_rad):.3f} deg")

print("\n========== AMPLITUDE PERFORMANCE ==========")
print(f"Amplitude jitter RMS    : {amp_jitter_db:.3f} dB")
print(f"Amplitude error p-p     : {amp_peak_to_peak_db:.3f} dB")


# ============================================================
# QUALITY JUDGEMENT
# ============================================================
print("\n========== JUDGEMENT ==========")

if len(bad_headers) == 0:
    print("SYNC                 : GOOD, no bad headers")
else:
    print("SYNC                 : BAD, bad headers exist")

if target_snr_db >= 25:
    print("TARGET SNR           : GOOD")
elif target_snr_db >= 15:
    print("TARGET SNR           : USABLE")
else:
    print("TARGET SNR           : WEAK, phase may be noisy")

if phase_jitter_deg < 5:
    print("DOPPLER PHASE        : VERY GOOD")
elif phase_jitter_deg < 15:
    print("DOPPLER PHASE        : USABLE")
elif phase_jitter_deg < 30:
    print("DOPPLER PHASE        : MARGINAL")
else:
    print("DOPPLER PHASE        : BAD")

if phase_jitter_deg < 5:
    print("SAR PHASE            : GOOD")
elif phase_jitter_deg < 10:
    print("SAR PHASE            : USABLE")
elif phase_jitter_deg < 20:
    print("SAR PHASE            : WEAK / MAY DEFOCUS")
else:
    print("SAR PHASE            : BAD / LIKELY DEFOCUSED")

if amp_jitter_db < 1:
    print("AMPLITUDE STABILITY  : GOOD")
elif amp_jitter_db < 2:
    print("AMPLITUDE STABILITY  : USABLE")
else:
    print("AMPLITUDE STABILITY  : UNSTABLE")


# ============================================================
# PLOTS
# ============================================================

# 1) Average range profile
plt.figure(figsize=(12, 5))
plt.plot(range_m[valid], 20 * np.log10(avg_mag[valid] + 1e-12))
plt.axvline(track_range_m, linestyle="--", label=f"Tracked bin {track_bin}, {track_range_m:.2f} m")
plt.title("Average Range Profile")
plt.xlabel("Range (m)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 2) Wrapped and unwrapped phase
plt.figure(figsize=(12, 5))
plt.plot(n, phase_wrapped, label="Wrapped phase")
plt.title("Tracked Bin Wrapped Phase")
plt.xlabel("Chirp index")
plt.ylabel("Phase (rad)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.plot(n, phase_unwrapped, label="Unwrapped phase")
plt.plot(n, phase_fit, label="Linear trend")
plt.title("Tracked Bin Unwrapped Phase")
plt.xlabel("Chirp index")
plt.ylabel("Phase (rad)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# 3) Phase error after detrending
plt.figure(figsize=(12, 5))
plt.plot(n, np.rad2deg(phase_error))
plt.title(f"Phase Error After Detrend | RMS = {phase_jitter_deg:.3f} deg")
plt.xlabel("Chirp index")
plt.ylabel("Phase error (deg)")
plt.grid(True)
plt.tight_layout()

# 4) Amplitude stability
plt.figure(figsize=(12, 5))
plt.plot(n, amp_db, label="Amplitude")
plt.plot(n, amp_fit_db, label="Linear trend")
plt.title("Tracked Bin Amplitude")
plt.xlabel("Chirp index")
plt.ylabel("Amplitude (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure(figsize=(12, 5))
plt.plot(n, amp_error_db)
plt.title(f"Amplitude Error After Detrend | RMS = {amp_jitter_db:.3f} dB")
plt.xlabel("Chirp index")
plt.ylabel("Amplitude error (dB)")
plt.grid(True)
plt.tight_layout()

# 5) Chirp-to-chirp phase step
plt.figure(figsize=(12, 5))
plt.plot(np.rad2deg(phase_step))
plt.title(f"Chirp-to-Chirp Phase Step | STD = {np.rad2deg(phase_step_std_rad):.3f} deg")
plt.xlabel("Chirp transition index")
plt.ylabel("Phase step (deg)")
plt.grid(True)
plt.tight_layout()

plt.show()