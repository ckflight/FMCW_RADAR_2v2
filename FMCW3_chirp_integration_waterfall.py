import numpy as np
import matplotlib.pyplot as plt
import struct

# =========================================================
# SETTINGS
# =========================================================

FILENAME = "record.bin"

INTEGRATION_CHIRPS = 128      # software CPI size: 32, 64, 128, etc.
BLOCK_STEP = 128              # 128 = no overlap, 1 = sliding integration

USE_LOG_MAX_DISTANCE = True # plots calcualted max distance by fs and hz/m
MAX_RANGE_DISPLAY = 125 # user selected range if USE_LOG_MAX_DISTANCE = False

REMOVE_DC = True
USE_WINDOW = True

INFO_SECTOR_SIZE = 512
HEADER = b"\xC8\xC8\xC8\xC8"

ADC_BITS = 12
ADC_FS = 2 ** (ADC_BITS - 1)

BYTES_PER_SAMPLE_PAIR = 4
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
    p["ADC_SELECT"], o = read_u32(info, o)
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

def decode_adc_block(chirp_bytes_list):
    block_a = []
    block_b = []

    for chirp_bytes in chirp_bytes_list:
        raw = np.frombuffer(chirp_bytes, dtype=">i2").astype(np.float32)

        adc_a = raw[0::2]
        adc_b = raw[1::2]

        block_a.append(adc_a)
        block_b.append(adc_b)

    return np.array(block_a), np.array(block_b)


# =========================================================
# FFT HELPERS
# =========================================================

def range_fft_block_dbfs(chirps_block):
    x = chirps_block.astype(np.float32)

    if REMOVE_DC:
        x = x - np.mean(x, axis=1, keepdims=True)

    if USE_WINDOW:
        window = np.hanning(x.shape[1]).astype(np.float32)
        coherent_gain = np.sum(window) / len(window)
        x = x * window
    else:
        coherent_gain = 1.0

    fft_data = np.fft.rfft(x, axis=1)

    mag = np.abs(fft_data)
    mag = mag / (x.shape[1] * coherent_gain / 2.0)

    mag_dbfs = 20 * np.log10((mag / ADC_FS) + 1e-15)

    return fft_data, mag_dbfs


def integrated_range_dbfs(chirps_fft):
    power = np.mean(np.abs(chirps_fft) ** 2, axis=0)

    amp_equiv = np.sqrt(power)
    amp_equiv = amp_equiv / (SAMPLES_PER_CHIRP / 2.0)

    dbfs = 20 * np.log10((amp_equiv / ADC_FS) + 1e-15)

    return dbfs


def range_doppler_db(chirps_fft):
    doppler_fft = np.fft.fft(chirps_fft, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)

    rd_db = 20 * np.log10(np.abs(doppler_fft) + 1e-15)

    return rd_db


# =========================================================
# READ FILE
# =========================================================

with open(FILENAME, "rb") as f:
    raw_all = f.read()

info = parse_info_sector(raw_all[:INFO_SECTOR_SIZE])
raw_data = raw_all[INFO_SECTOR_SIZE:]

SAMPLES_PER_CHIRP = int(info["NUMBER_OF_SAMPLES"])
FS = float(info["SAMPLING_FREQUENCY"])
SWEEP_TIME = float(info["SWEEP_TIME"])
SWEEP_BW = float(info["SWEEP_BW"])

slope = SWEEP_BW / SWEEP_TIME

bytes_per_chirp = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE_PAIR

# =========================================================
# PRINT INFO SECTOR
# =========================================================
print("\n========== LOG INFO ==========")
print(f"Version              : {info['VERSION']}")
print(f"Record time          : {info['RECORD_TIME']} s")

print("\n----- Chirp -----")
print(f"Sweep time           : {info['SWEEP_TIME'] * 1e6:.0f} us")
print(f"Sweep gap            : {info['SWEEP_GAP'] * 1e6:.0f} us")
print(f"Sweep start          : {info['SWEEP_START'] / 1e9:.3f} GHz")
print(f"Sweep bandwidth      : {info['SWEEP_BW'] / 1e6:.1f} MHz")
print(f"Hz per meter         : {info['HZ_PER_M']:.2f} Hz/m")

max_beat_frequency = info['SAMPLING_FREQUENCY'] / 2.0
max_distance = max_beat_frequency / info['HZ_PER_M']

print(f"Max distance         : {max_distance:.2f} m")

print("\n----- Sampling -----")
print(f"Sampling frequency   : {info['SAMPLING_FREQUENCY'] / 1e6:.3f} MHz")
print(f"Samples per chirp    : {info['NUMBER_OF_SAMPLES']}")
print(f"ADC resolution       : {info['ADC_RESOLUTION']} bit")
print(f"Sample averaging     : {info['SAMPLE_AVERAGING']}")

print("\n----- Modes -----")
print(f"Test mux             : {info['TEST_MUX']}")
print(f"Gain                 : {info['GAIN']}")
print(f"Sweep type           : {info['SWEEP_TYPE']}")
print(f"Data log             : {info['DATA_LOG']}")
print(f"ADC select           : {info['ADC_SELECT']}")
print(f"PA mode              : {info['PA_MODE']}")
print(f"FIR enable           : {info['FIR_ENABLE']}")
print(f"Send data type       : {info['SEND_DATA_TYPE']}")

print("\n----- File Layout -----")
print(f"Info sector size     : {info['INFO_SECTOR_SIZE']} bytes")
print(f"Data start offset    : {info['DATA_START_OFFSET']} bytes")
print("==============================\n")

if USE_LOG_MAX_DISTANCE == True:
    MAX_RANGE_DISPLAY = int(max_distance)



# =========================================================
# FAST HEADER SEARCH
# =========================================================

raw_u8 = np.frombuffer(raw_data, dtype=np.uint8)

header = np.frombuffer(HEADER, dtype=np.uint8)
header_len = len(header)

matches = np.where(
    (raw_u8[:-3] == header[0]) &
    (raw_u8[1:-2] == header[1]) &
    (raw_u8[2:-1] == header[2]) &
    (raw_u8[3:] == header[3])
)[0]

chirps = []

for idx in matches:
    start = idx + header_len
    end = start + bytes_per_chirp

    if end <= len(raw_data):
        chirps.append(raw_data[start:end])

num_chirps = len(chirps)

if num_chirps < INTEGRATION_CHIRPS:
    raise RuntimeError("Not enough chirps for selected integration length")

print("Valid chirps  :", num_chirps)


# =========================================================
# RANGE AXIS
# =========================================================

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
range_m = freq_hz * C / (2.0 * slope)

range_mask = range_m <= MAX_RANGE_DISPLAY
range_limited = range_m[range_mask]


# =========================================================
# WATERFALL SIZE
# =========================================================

num_blocks = 1 + (num_chirps - INTEGRATION_CHIRPS) // BLOCK_STEP

waterfall_a = np.full(
    (num_blocks, len(range_limited)),
    -140,
    dtype=np.float32
)
waterfall_b = np.full(
    (num_blocks, len(range_limited)),
    -140,
    dtype=np.float32
)

# =========================================================
# PLOTS
# =========================================================

plt.ion()

fig, axes = plt.subplots(4, 1, figsize=(12, 11))
fig.subplots_adjust(hspace=0.45)

ax_range_a = axes[0]
ax_range_b = axes[1]
ax_waterfall = axes[2]
ax_rd = axes[3]

line_a, = ax_range_a.plot([], [])
line_b, = ax_range_b.plot([], [])

img_waterfall = ax_waterfall.imshow(
    np.zeros_like(waterfall_a),
    aspect="auto",
    origin="lower",
    extent=[range_limited[0], range_limited[-1], 0, num_blocks]
)

img_rd = ax_rd.imshow(
    np.zeros((INTEGRATION_CHIRPS, len(range_limited))),
    aspect="auto",
    origin="lower",
    extent=[range_limited[0], range_limited[-1], -INTEGRATION_CHIRPS / 2, INTEGRATION_CHIRPS / 2]
)

ax_range_a.set_title("ADC A Integrated Range Profile")
ax_range_b.set_title("ADC B Integrated Range Profile")
ax_waterfall.set_title("ADC A Waterfall")
ax_rd.set_title("ADC A Range-Doppler Map")

ax_range_a.set_ylabel("dBFS")
ax_range_b.set_ylabel("dBFS")
ax_waterfall.set_ylabel("Software CPI Index")
ax_rd.set_ylabel("Doppler Bin")

ax_rd.set_xlabel("Range (m)")

for ax in axes:
    ax.grid(True)

ax_range_a.set_xlim(0, MAX_RANGE_DISPLAY)
ax_range_b.set_xlim(0, MAX_RANGE_DISPLAY)


# =========================================================
# MAIN LOOP
# =========================================================

for block_idx in range(num_blocks):

    chirp_start = block_idx * BLOCK_STEP
    chirp_end = chirp_start + INTEGRATION_CHIRPS

    chirps_block_bytes = chirps[chirp_start:chirp_end]

    adc_a_block, adc_b_block = decode_adc_block(chirps_block_bytes)

    chirps_fft_a, mag_dbfs_a = range_fft_block_dbfs(adc_a_block)
    chirps_fft_b, mag_dbfs_b = range_fft_block_dbfs(adc_b_block)

    integrated_a = integrated_range_dbfs(chirps_fft_a)
    integrated_b = integrated_range_dbfs(chirps_fft_b)

    integrated_a_limited = integrated_a[range_mask]
    integrated_b_limited = integrated_b[range_mask]

    waterfall_a[block_idx, :] = integrated_a_limited
    waterfall_b[block_idx, :] = integrated_b_limited

    rd_a = range_doppler_db(chirps_fft_a)
    rd_a_limited = rd_a[:, range_mask]

    line_a.set_data(range_limited, integrated_a_limited)
    line_b.set_data(range_limited, integrated_b_limited)

    ax_range_a.set_ylim(np.max(integrated_a_limited) - 80, np.max(integrated_a_limited) + 5)
    ax_range_b.set_ylim(np.max(integrated_b_limited) - 80, np.max(integrated_b_limited) + 5)

    img_waterfall.set_data(waterfall_a)
    img_waterfall.set_clim(np.max(waterfall_a) - 50, np.max(waterfall_a))

    img_rd.set_data(rd_a_limited)
    img_rd.set_clim(np.max(rd_a_limited) - 50, np.max(rd_a_limited))

    fig.suptitle(
        f"Software CPI {block_idx + 1}/{num_blocks} | "
        f"Chirps {chirp_start} to {chirp_end - 1} | "
        f"N = {INTEGRATION_CHIRPS}"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.001)

plt.ioff()
plt.show()