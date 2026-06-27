import numpy as np
import matplotlib.pyplot as plt
import struct

# =========================================================
# USER SETTINGS
# =========================================================

FILENAME = "record.bin"#"fmcw3_bin_files/250khz_900mhz.bin"

START_CHIRP = 0
END_CHIRP = None
CHIRP_STEP = 100
FRAME_DELAY = 0.001

REMOVE_DC = False
USE_WINDOW = True

# =========================================================

HEADER = b"\xC8\xC8\xC8\xC8"
INFO_SECTOR_SIZE = 512
BYTES_PER_SAMPLE_PAIR = 4

ADC_BITS = 12
ADC_FS = 2 ** (ADC_BITS - 1)

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

    raw = np.frombuffer(chirp_bytes, dtype=">i2").astype(np.int32) # big endian. 12 bit signed adc is resized as 16bit. fir is also 16 bit in out

    adc_a = raw[0::2]
    adc_b = raw[1::2]

    return adc_a, adc_b


# =========================================================
# FFT
# =========================================================

def calculate_fft_dbfs(signal, fs):

    x = signal.astype(np.float32)

    if REMOVE_DC:
        x = x - np.mean(x)

    if USE_WINDOW:
        window = np.hanning(len(x))
        coherent_gain = np.sum(window) / len(window)
        x = x * window
    else:
        coherent_gain = 1.0

    fft_data = np.fft.rfft(x)

    freq = np.fft.rfftfreq(len(x), d=1.0 / fs)

    mag = np.abs(fft_data)
    mag = mag / (len(x) * coherent_gain / 2)

    mag_dbfs = 20 * np.log10(mag / ADC_FS + 1e-12)

    return freq, mag_dbfs


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

bytes_per_chirp = samples_per_chirp * BYTES_PER_SAMPLE_PAIR

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
print(f"PA mode              : {info['PA_MODE']}")
print(f"FIR enable           : {info['FIR_ENABLE']}")
print(f"Send data type       : {info['SEND_DATA_TYPE']}")

print("\n----- File Layout -----")
print(f"Info sector size     : {info['INFO_SECTOR_SIZE']} bytes")
print(f"Data start offset    : {info['DATA_START_OFFSET']} bytes")
print("==============================\n")


# =========================================================
# FIND CHIRPS
# =========================================================

chirps = []

idx = 0

while True:

    idx = raw_data.find(HEADER, idx)

    if idx < 0:
        break

    start = idx + len(HEADER)
    end = start + bytes_per_chirp

    if end <= len(raw_data):
        chirps.append(raw_data[start:end])

    idx += len(HEADER)

if len(chirps) == 0:
    raise RuntimeError("No chirps found")

num_chirps = len(chirps)

if END_CHIRP is None:
    END_CHIRP = num_chirps

print("Valid chirps  :", num_chirps)

# =========================================================
# INITIAL DATA
# =========================================================

adc_a, adc_b = decode_adc(chirps[0])

freq_a, mag_a = calculate_fft_dbfs(adc_a, fs)
freq_b, mag_b = calculate_fft_dbfs(adc_b, fs)

range_a = freq_a * C / (2 * slope)
range_b = freq_b * C / (2 * slope)

x_sample = np.arange(samples_per_chirp)

# =========================================================
# PLOTS
# =========================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

ax_a_time = axes[0]
ax_b_time = axes[1]
ax_a_fft = axes[2]
ax_b_fft = axes[3]

line_a_time, = ax_a_time.plot(x_sample, adc_a)
line_b_time, = ax_b_time.plot(x_sample, adc_b)

line_a_fft, = ax_a_fft.plot(range_a, mag_a)
line_b_fft, = ax_b_fft.plot(range_b, mag_b)

ax_a_time.set_title("ADC A Time Domain")
ax_b_time.set_title("ADC B Time Domain")

ax_a_fft.set_title("ADC A FFT")
ax_b_fft.set_title("ADC B FFT")

ax_a_time.set_ylabel("ADC Code")
ax_b_time.set_ylabel("ADC Code")

ax_a_fft.set_ylabel("dBFS")
ax_b_fft.set_ylabel("dBFS")

ax_b_time.set_xlabel("Sample")
ax_b_fft.set_xlabel("Range (m)")

ax_a_time.set_ylim(-2048, 2047)
ax_b_time.set_ylim(-2048, 2047)

ax_a_fft.set_xlim(0, range_a.max())
ax_b_fft.set_xlim(0, range_b.max())

for ax in axes:
    ax.grid(True)

# =========================================================
# UPDATE LOOP
# =========================================================

plt.ion()

for chirp_idx in range(START_CHIRP, END_CHIRP, CHIRP_STEP):

    adc_a, adc_b = decode_adc(chirps[chirp_idx])

    freq_a, mag_a = calculate_fft_dbfs(adc_a, fs)
    freq_b, mag_b = calculate_fft_dbfs(adc_b, fs)

    range_a = freq_a * C / (2 * slope)
    range_b = freq_b * C / (2 * slope)

    line_a_time.set_ydata(adc_a)
    line_b_time.set_ydata(adc_b)

    line_a_fft.set_xdata(range_a)
    line_a_fft.set_ydata(mag_a)

    line_b_fft.set_xdata(range_b)
    line_b_fft.set_ydata(mag_b)

    ax_a_fft.set_ylim(mag_a.max() - 80, mag_a.max() + 5)
    ax_b_fft.set_ylim(mag_b.max() - 80, mag_b.max() + 5)

    fig.suptitle(f"Chirp {chirp_idx + 1}/{num_chirps}")

    fig.tight_layout()

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()