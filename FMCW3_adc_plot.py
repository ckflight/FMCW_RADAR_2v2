import numpy as np
import matplotlib.pyplot as plt
import struct

# ================= USER SETTINGS =================

FILENAME = "record.bin"

PLOT_ADC = 1          # 1 = plot ADC time-domain
PLOT_FFT = 1          # 1 = plot FFT/range-domain

START_CHIRP = 0
END_CHIRP = None
CHIRP_STEP = 1
FRAME_DELAY = 0.001

CONVERT_TO_VOLTAGE = 0
ADC_FULL_SCALE_VOLTAGE = 3.0

REMOVE_DC = False
USE_WINDOW = True

ADC_ENDIAN = ">i2"  # < looks like a waveform but > is the correct format

# =================================================

HEADER = b"\xC8\xC8\xC8\xC8"
INFO_SECTOR_SIZE = 512
BYTES_PER_SAMPLE_PAIR = 4
C = 3e8


def read_u32(buf, offset):
    return struct.unpack_from("<I", buf, offset)[0], offset + 4


def read_f32(buf, offset):
    return struct.unpack_from("<f", buf, offset)[0], offset + 4


def parse_info_sector(info):
    if info[0:4] != b"FMCW":
        raise RuntimeError("Invalid info sector magic")

    o = 4
    parsed = {}

    parsed["VERSION"], o = read_u32(info, o)
    parsed["SWEEP_TIME"], o = read_f32(info, o)
    parsed["SWEEP_GAP"], o = read_f32(info, o)
    parsed["RECORD_TIME"], o = read_u32(info, o)

    parsed["SAMPLING_FREQUENCY"], o = read_u32(info, o)
    parsed["NUMBER_OF_SAMPLES"], o = read_u32(info, o)

    parsed["SWEEP_START"], o = read_f32(info, o)
    parsed["SWEEP_BW"], o = read_f32(info, o)

    # Skip unused config words
    for _ in range(10):
        _, o = read_u32(info, o)

    parsed["HZ_PER_M"], o = read_f32(info, o)
    parsed["INFO_SECTOR_SIZE"], o = read_u32(info, o)
    parsed["DATA_START_BYTE"], o = read_u32(info, o)

    return parsed


def print_info(info):
    print("\nINFO")
    print("----")
    print("Sweep time        :", f"{info['SWEEP_TIME'] * 1e6:.0f} us")
    print("Sweep gap         :", f"{info['SWEEP_GAP'] * 1e6:.0f} us")
    print("Sampling freq     :", f"{info['SAMPLING_FREQUENCY'] / 1e6:.3f} MHz")
    print("Samples/chirp     :", info["NUMBER_OF_SAMPLES"])
    print("Sweep start       :", f"{info['SWEEP_START'] / 1e6:.0f} MHz")
    print("Sweep BW          :", f"{info['SWEEP_BW'] / 1e6:.0f} MHz")


def calculate_fft(signal, fs):
    signal = signal.astype(np.float32)

    if REMOVE_DC:
        signal = signal - np.mean(signal)

    if USE_WINDOW:
        signal = signal * np.hanning(len(signal))

    freq = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    fft_data = np.fft.rfft(signal)
    mag_db = 20 * np.log10(np.abs(fft_data) + 1e-12)

    return freq, mag_db


# ================= READ FILE =================

with open(FILENAME, "rb") as f:
    raw_all = f.read()

info_raw = raw_all[:INFO_SECTOR_SIZE]
raw_data = raw_all[INFO_SECTOR_SIZE:]

info = parse_info_sector(info_raw)
print_info(info)

samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])
fs = float(info["SAMPLING_FREQUENCY"])
slope = info["SWEEP_BW"] / info["SWEEP_TIME"]

bytes_per_chirp = samples_per_chirp * BYTES_PER_SAMPLE_PAIR

# ================= FIND CHIRPS =================

header_positions = []
idx = 0

while True:
    idx = raw_data.find(HEADER, idx)

    if idx < 0:
        break

    header_positions.append(idx)
    idx += len(HEADER)

print("\nHeaders found      :", len(header_positions))

chirp_payloads = []

for h in header_positions:
    start = h + len(HEADER)
    end = start + bytes_per_chirp

    if end <= len(raw_data):
        chirp_payloads.append(raw_data[start:end])

if len(chirp_payloads) == 0:
    raise RuntimeError("No valid chirps found")

print("Valid chirps       :", len(chirp_payloads))

# ================= DECODE ADC =================

adc_a_chirps = []
adc_b_chirps = []

adc_max = 32768.0

for chirp in chirp_payloads:
    samples = np.frombuffer(chirp, dtype=ADC_ENDIAN)

    # Same logic as plot_log.py:
    # ch1 = l[::2]
    # ch2 = l[1::2]
    adc_a = samples[0::2]
    adc_b = samples[1::2]

    if CONVERT_TO_VOLTAGE == 1:
        adc_a_v = adc_a.astype(np.float32) / adc_max * ADC_FULL_SCALE_VOLTAGE
        adc_b_v = adc_b.astype(np.float32) / adc_max * ADC_FULL_SCALE_VOLTAGE
    
        adc_a_chirps.append(adc_a_v)
        adc_b_chirps.append(adc_b_v)

    else:
        adc_a_chirps.append(adc_a)
        adc_b_chirps.append(adc_b)

adc_a_chirps = np.array(adc_a_chirps)
adc_b_chirps = np.array(adc_b_chirps)

num_chirps = len(adc_a_chirps)

if END_CHIRP is None:
    END_CHIRP = num_chirps

print("Samples/channel    :", adc_a_chirps.shape[1])

# ================= CREATE PLOTS =================

plot_count = 0

if PLOT_ADC:
    plot_count += 2

if PLOT_FFT:
    plot_count += 2

if plot_count == 0:
    raise RuntimeError("Enable PLOT_ADC or PLOT_FFT")

fig, axes = plt.subplots(plot_count, 1, figsize=(12, 2.7 * plot_count))

if plot_count == 1:
    axes = [axes]

ax_index = 0

x_sample = np.arange(samples_per_chirp)

if PLOT_ADC:
    ax_adc_a = axes[ax_index]
    ax_index += 1

    ax_adc_b = axes[ax_index]
    ax_index += 1

    line_adc_a, = ax_adc_a.plot(x_sample, adc_a_chirps[0])
    line_adc_b, = ax_adc_b.plot(x_sample, adc_b_chirps[0])

    ax_adc_a.set_title("ADC A Time Domain")
    ax_adc_b.set_title("ADC B Time Domain")

    ax_adc_a.set_ylabel("Voltage (V)")
    ax_adc_b.set_ylabel("Voltage (V)")
    ax_adc_b.set_xlabel("Sample")

    ax_adc_a.grid(True)
    ax_adc_b.grid(True)

    ax_adc_a.set_ylim(adc_a_chirps.min() - 0.05, adc_a_chirps.max() + 0.05)
    ax_adc_b.set_ylim(adc_b_chirps.min() - 0.05, adc_b_chirps.max() + 0.05)

if PLOT_FFT:
    ax_fft_a = axes[ax_index]
    ax_index += 1

    ax_fft_b = axes[ax_index]
    ax_index += 1

    freq_a, mag_a = calculate_fft(adc_a_chirps[0], fs)
    freq_b, mag_b = calculate_fft(adc_b_chirps[0], fs)

    range_a = freq_a * C / (2 * slope)
    range_b = freq_b * C / (2 * slope)

    line_fft_a, = ax_fft_a.plot(range_a, mag_a)
    line_fft_b, = ax_fft_b.plot(range_b, mag_b)

    ax_fft_a.set_title("ADC A Range FFT")
    ax_fft_b.set_title("ADC B Range FFT")

    ax_fft_a.set_ylabel("Magnitude (dB)")
    ax_fft_b.set_ylabel("Magnitude (dB)")
    ax_fft_b.set_xlabel("Range (m)")

    ax_fft_a.grid(True)
    ax_fft_b.grid(True)

    ax_fft_a.set_xlim(0, range_a.max())
    ax_fft_b.set_xlim(0, range_b.max())

# ================= UPDATE LOOP =================

plt.ion()

for chirp_idx in range(START_CHIRP, END_CHIRP, CHIRP_STEP):
    adc_a = adc_a_chirps[chirp_idx]
    adc_b = adc_b_chirps[chirp_idx]

    if PLOT_ADC:
        line_adc_a.set_ydata(adc_a)
        line_adc_b.set_ydata(adc_b)

    if PLOT_FFT:
        freq_a, mag_a = calculate_fft(adc_a, fs)
        freq_b, mag_b = calculate_fft(adc_b, fs)

        range_a = freq_a * C / (2 * slope)
        range_b = freq_b * C / (2 * slope)

        line_fft_a.set_xdata(range_a)
        line_fft_a.set_ydata(mag_a)

        line_fft_b.set_xdata(range_b)
        line_fft_b.set_ydata(mag_b)

        ax_fft_a.set_ylim(mag_a.min() - 5, mag_a.max() + 5)
        ax_fft_b.set_ylim(mag_b.min() - 5, mag_b.max() + 5)

    fig.suptitle(f"Chirp {chirp_idx + 1}/{num_chirps}")
    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()