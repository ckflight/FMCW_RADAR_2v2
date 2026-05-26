import numpy as np
import matplotlib.pyplot as plt
import struct

# =========================================================
# USER SETTINGS
# =========================================================

FILENAME = "fmcw3_bin_files/250khz_900mhz.bin"

CHANNEL_SELECT = "A"       # "A" or "B"

CHIRPS_PER_CPI = 128       # set this manually for FMCW3
MAX_RANGE_TO_SHOW = 100

AUTO_PICK_BIN = True
FORCED_BIN = 54
IGNORE_FIRST_BINS = 5
AVG_CHIRPS_FOR_BIN_PICK = 256

DISPLAY_EVERY_N_CPI = 1
FRAME_DELAY = 0.001

REMOVE_DC = True
USE_WINDOW = True

# =========================================================

HEADER = b"\xC8\xC8\xC8\xC8"
INFO_SECTOR_SIZE = 512
BYTES_PER_SAMPLE_PAIR = 4

ADC_BITS_DEFAULT = 12
C = 3e8


# =========================================================
# INFO SECTOR
# =========================================================

def read_u32(buf, offset):
    return struct.unpack_from("<I", buf, offset)[0], offset + 4


def read_f32(buf, offset):
    return struct.unpack_from("<f", buf, offset)[0], offset + 4


def parse_info_sector(info):
    if info[0:4] != b"FMCW":
        raise RuntimeError("Invalid info sector")

    o = 4
    p = {}

    p["VERSION"], o = read_u32(info, o)

    p["SWEEP_TIME"], o = read_f32(info, o)
    p["SWEEP_GAP"], o = read_f32(info, o)
    p["RECORD_TIME"], o = read_u32(info, o)

    p["SAMPLING_FREQUENCY"], o = read_u32(info, o)
    p["NUMBER_OF_SAMPLES"], o = read_u32(info, o)

    p["SWEEP_START"], o = read_f32(info, o)
    p["SWEEP_BW"], o = read_f32(info, o)

    p["TEST_MUX"], o = read_u32(info, o)
    p["GAIN"], o = read_u32(info, o)
    p["SWEEP_TYPE"], o = read_u32(info, o)
    p["DATA_LOG"], o = read_u32(info, o)
    p["PA_MODE"], o = read_u32(info, o)
    p["FIR_ENABLE"], o = read_u32(info, o)
    p["SEND_DATA_TYPE"], o = read_u32(info, o)
    p["ADC_RESOLUTION"], o = read_u32(info, o)
    p["SAMPLE_AVERAGING"], o = read_u32(info, o)

    p["HZ_PER_M"], o = read_f32(info, o)

    p["INFO_SECTOR_SIZE"], o = read_u32(info, o)
    p["DATA_START_OFFSET"], o = read_u32(info, o)

    return p


# =========================================================
# ADC DECODE
# =========================================================

def decode_adc(chirp_bytes):
    raw = np.frombuffer(chirp_bytes, dtype=">i2").astype(np.int32)

    adc_a = raw[0::2]
    adc_b = raw[1::2]

    return adc_a, adc_b


# =========================================================
# READ FILE
# =========================================================

with open(FILENAME, "rb") as f:
    raw_all = f.read()

info = parse_info_sector(raw_all[:INFO_SECTOR_SIZE])
raw_data = raw_all[INFO_SECTOR_SIZE:]

samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])
fs = float(info["SAMPLING_FREQUENCY"])
slope = float(info["SWEEP_BW"]) / float(info["SWEEP_TIME"])

adc_bits = int(info["ADC_RESOLUTION"])
if adc_bits <= 0:
    adc_bits = ADC_BITS_DEFAULT

adc_fs = 2 ** (adc_bits - 1)

bytes_per_chirp = samples_per_chirp * BYTES_PER_SAMPLE_PAIR

print("\n========== LOG INFO ==========")
print(f"Version              : {info['VERSION']}")
print(f"Record time          : {info['RECORD_TIME']} s")
print(f"Sweep time           : {info['SWEEP_TIME'] * 1e6:.0f} us")
print(f"Sweep gap            : {info['SWEEP_GAP'] * 1e6:.0f} us")
print(f"Sweep start          : {info['SWEEP_START'] / 1e9:.3f} GHz")
print(f"Sweep bandwidth      : {info['SWEEP_BW'] / 1e6:.1f} MHz")
print(f"Hz per meter         : {info['HZ_PER_M']:.2f} Hz/m")
print(f"Sampling frequency   : {fs / 1e6:.3f} MHz")
print(f"Samples per chirp    : {samples_per_chirp}")
print(f"ADC resolution       : {adc_bits} bit")
print(f"Selected channel     : ADC {CHANNEL_SELECT}")
print(f"CHIRPS_PER_CPI       : {CHIRPS_PER_CPI}")
print("==============================\n")


# =========================================================
# FIND CHIRPS
# =========================================================

chirps_bytes = []

idx = 0

while True:
    idx = raw_data.find(HEADER, idx)

    if idx < 0:
        break

    start = idx + len(HEADER)
    end = start + bytes_per_chirp

    if end <= len(raw_data):
        chirps_bytes.append(raw_data[start:end])

    idx += len(HEADER)

if len(chirps_bytes) == 0:
    raise RuntimeError("No chirps found")

num_chirps = len(chirps_bytes)

print("Valid chirps:", num_chirps)

# =========================================================
# DECODE SELECTED ADC CHANNEL
# =========================================================

chirps = np.zeros((num_chirps, samples_per_chirp), dtype=np.float32)

for i, chirp_bytes in enumerate(chirps_bytes):
    adc_a, adc_b = decode_adc(chirp_bytes)

    if CHANNEL_SELECT.upper() == "A":
        chirps[i, :] = adc_a.astype(np.float32)
    else:
        chirps[i, :] = adc_b.astype(np.float32)

# =========================================================
# KEEP ONLY FULL CPIs
# =========================================================

num_cpis = num_chirps // CHIRPS_PER_CPI
num_chirps_used = num_cpis * CHIRPS_PER_CPI

if num_cpis == 0:
    raise RuntimeError("Not enough chirps for one full CPI")

chirps = chirps[:num_chirps_used]
chirps_3d = chirps.reshape(num_cpis, CHIRPS_PER_CPI, samples_per_chirp)

print("Num CPIs      :", num_cpis)
print("Chirps used   :", num_chirps_used)

# =========================================================
# FFT AXES
# =========================================================

freq_hz = np.fft.rfftfreq(samples_per_chirp, d=1.0 / fs)

if info["HZ_PER_M"] > 0:
    range_m = freq_hz / info["HZ_PER_M"]
else:
    range_m = freq_hz * C / (2 * slope)

# =========================================================
# COMPLEX RANGE FFT CUBE
# =========================================================

print("\nComputing complex FFTs...")

x = chirps_3d.copy()

if REMOVE_DC:
    x = x - np.mean(x, axis=2, keepdims=True)

if USE_WINDOW:
    w = np.hanning(samples_per_chirp).astype(np.float32)
    x = x * w[None, None, :]

range_fft = np.fft.rfft(x, axis=2)

# =========================================================
# AUTO PICK STRONG RANGE BIN
# =========================================================

flat_fft = range_fft.reshape(-1, range_fft.shape[2])
avg_count = min(AVG_CHIRPS_FOR_BIN_PICK, flat_fft.shape[0])

avg_mag = np.mean(np.abs(flat_fft[:avg_count]), axis=0)

valid = np.where(range_m <= MAX_RANGE_TO_SHOW)[0]
valid = valid[valid >= IGNORE_FIRST_BINS]

if len(valid) == 0:
    raise ValueError("No valid range bins")

if AUTO_PICK_BIN:
    track_bin = valid[np.argmax(avg_mag[valid])]
else:
    track_bin = FORCED_BIN

track_range_m = range_m[track_bin]
track_freq_hz = freq_hz[track_bin]

print("\n----- TRACKED BIN -----")
print(f"track_bin    : {track_bin}")
print(f"track_freq   : {track_freq_hz / 1e3:.2f} kHz")
print(f"track_range  : {track_range_m:.2f} m")

# =========================================================
# PHASE EXTRACTION
# =========================================================

tracked = range_fft[:, :, track_bin]

phase_chirp_wrapped = np.angle(tracked)
phase_chirp_unwrapped = np.unwrap(phase_chirp_wrapped, axis=1)

cpi_complex = np.mean(tracked, axis=1)

phase_cpi_wrapped = np.angle(cpi_complex)
phase_cpi_unwrapped = np.unwrap(phase_cpi_wrapped)

chirp_idx = np.arange(CHIRPS_PER_CPI)
cpi_idx = np.arange(num_cpis)

# =========================================================
# DISPLAY
# =========================================================

plt.ion()

fig, ax = plt.subplots(2, 2, figsize=(14, 8))

line_cw, = ax[0, 0].plot([], [], lw=1.0)
line_cu, = ax[1, 0].plot([], [], lw=1.0)

line_pw, = ax[0, 1].plot([], [], lw=1.0)
line_pu, = ax[1, 1].plot([], [], lw=1.0)

ax[0, 0].set_title("Chirp-to-Chirp Phase (wrapped)")
ax[1, 0].set_title("Chirp-to-Chirp Phase (unwrapped)")
ax[0, 1].set_title("CPI-to-CPI Phase (wrapped)")
ax[1, 1].set_title("CPI-to-CPI Phase (unwrapped)")

ax[0, 0].set_xlabel("Chirp index in CPI")
ax[1, 0].set_xlabel("Chirp index in CPI")
ax[0, 1].set_xlabel("CPI index")
ax[1, 1].set_xlabel("CPI index")

ax[0, 0].set_ylabel("Phase (rad)")
ax[1, 0].set_ylabel("Phase (rad)")
ax[0, 1].set_ylabel("Phase (rad)")
ax[1, 1].set_ylabel("Phase (rad)")

for a in ax.flatten():
    a.grid(True)

ax[0, 0].set_xlim(0, CHIRPS_PER_CPI - 1)
ax[1, 0].set_xlim(0, CHIRPS_PER_CPI - 1)
ax[0, 1].set_xlim(0, num_cpis - 1)
ax[1, 1].set_xlim(0, num_cpis - 1)

ax[0, 0].set_ylim(-np.pi - 0.2, np.pi + 0.2)
ax[0, 1].set_ylim(-np.pi - 0.2, np.pi + 0.2)

chirp_unwrap_min = np.min(phase_chirp_unwrapped)
chirp_unwrap_max = np.max(phase_chirp_unwrapped)

if np.isclose(chirp_unwrap_min, chirp_unwrap_max):
    chirp_unwrap_min -= 1
    chirp_unwrap_max += 1

ax[1, 0].set_ylim(chirp_unwrap_min - 0.2, chirp_unwrap_max + 0.2)

cpi_unwrap_min = np.min(phase_cpi_unwrapped)
cpi_unwrap_max = np.max(phase_cpi_unwrapped)

if np.isclose(cpi_unwrap_min, cpi_unwrap_max):
    cpi_unwrap_min -= 1
    cpi_unwrap_max += 1

ax[1, 1].set_ylim(cpi_unwrap_min - 0.2, cpi_unwrap_max + 0.2)

fig.suptitle("FMCW3 Phase Stability Check", fontsize=14)
fig.tight_layout()

# =========================================================
# LIVE PLAYBACK
# =========================================================

for cpi_i in range(0, num_cpis, DISPLAY_EVERY_N_CPI):

    y_left_wrapped = phase_chirp_wrapped[cpi_i]
    y_left_unwrapped = phase_chirp_unwrapped[cpi_i]

    line_cw.set_data(chirp_idx, y_left_wrapped)
    line_cu.set_data(chirp_idx, y_left_unwrapped)

    x_right = cpi_idx[:cpi_i + 1]
    y_right_wrapped = phase_cpi_wrapped[:cpi_i + 1]
    y_right_unwrapped = phase_cpi_unwrapped[:cpi_i + 1]

    line_pw.set_data(x_right, y_right_wrapped)
    line_pu.set_data(x_right, y_right_unwrapped)

    fig.suptitle(
        f"FMCW3 Phase Stability Check | ADC {CHANNEL_SELECT.upper()} | "
        f"CPI {cpi_i + 1}/{num_cpis} | "
        f"bin={track_bin} | range={track_range_m:.2f} m",
        fontsize=14
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()