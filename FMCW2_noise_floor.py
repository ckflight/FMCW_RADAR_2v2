import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================

FILENAME = "Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_41_58_parking_lot_sar.txt"

START_CHIRP = 0
END_CHIRP = None
CHIRP_STEP = 1
FRAME_DELAY = 0.001

REMOVE_DC = False
USE_WINDOW = True

IGNORE_FIRST_BINS = 5
NOISE_PERCENTILE = 20

C = 3e8


# =========================================================
# TXT INFO PARSER
# =========================================================

def read_int_line(lines, idx, name):
    value = int(lines[idx].strip())
    return value, idx + 1


def parse_txt_info(lines):
    p = {}
    idx = 0

    p["RECORD_COUNTER"], idx = read_int_line(lines, idx, "RECORD_COUNTER")
    p["RECORD_TIME"], idx = read_int_line(lines, idx, "RECORD_TIME")

    p["SWEEP_TIME_US"], idx = read_int_line(lines, idx, "SWEEP_TIME_US")
    p["SWEEP_GAP_US"], idx = read_int_line(lines, idx, "SWEEP_GAP_US")

    p["SWEEP_TIME"] = p["SWEEP_TIME_US"] * 1e-6
    p["SWEEP_GAP"] = p["SWEEP_GAP_US"] * 1e-6

    p["SWEEP_START"], idx = read_int_line(lines, idx, "SWEEP_START")
    p["SWEEP_BW"], idx = read_int_line(lines, idx, "SWEEP_BW")

    p["SAMPLING_FREQUENCY"], idx = read_int_line(lines, idx, "SAMPLING_FREQUENCY")
    p["NUMBER_OF_SAMPLES"], idx = read_int_line(lines, idx, "NUMBER_OF_SAMPLES")

    p["TX_MODE"], idx = read_int_line(lines, idx, "TX_MODE")
    p["TX_POWER_DBM"], idx = read_int_line(lines, idx, "TX_POWER_DBM")
    p["TX_POWER_DBM_VOLTAGE"], idx = read_int_line(lines, idx, "TX_POWER_DBM_VOLTAGE")

    p["HZ_PER_M"], idx = read_int_line(lines, idx, "HZ_PER_M")

    p["DATA_LOG"], idx = read_int_line(lines, idx, "DATA_LOG")
    p["ADC_SELECT"], idx = read_int_line(lines, idx, "ADC_SELECT")
    p["USB_DATA_TYPE"], idx = read_int_line(lines, idx, "USB_DATA_TYPE")
    p["ADC_RESOLUTION"], idx = read_int_line(lines, idx, "ADC_RESOLUTION")
    p["PHASE_DISTANCE"], idx = read_int_line(lines, idx, "PHASE_DISTANCE")

    # Newer record code may have CPI_CHIRP before date.
    # Older txt files may directly have date after PHASE_DISTANCE.
    maybe_next = lines[idx].strip()

    try:
        p["CPI_CHIRP"] = int(maybe_next)
        idx += 1
        p["RECORD_DATE"] = lines[idx].strip()
        idx += 1
    except ValueError:
        p["CPI_CHIRP"] = 1
        p["RECORD_DATE"] = maybe_next
        idx += 1

    p["DATA_START_LINE"] = idx

    return p


# =========================================================
# USB TXT HEX DECODE
# =========================================================

def decode_usb_hex_line(hex_line, info):
    hex_line = hex_line.strip()

    if len(hex_line) == 0:
        return None

    b = bytes.fromhex(hex_line)

    adc_select = int(info["ADC_SELECT"])
    usb_data_type = int(info["USB_DATA_TYPE"])
    adc_bits = int(info["ADC_RESOLUTION"])

    samples = []

    idx = 0

    if adc_select == 1:
        # MAX1426 style packed 10-bit:
        # current_sample_16bit = ((byte0 & 0xF) << 6) | (byte1 & 0x3F)
        while idx + 1 < len(b):
            s = ((b[idx] & 0x0F) << 6) | (b[idx + 1] & 0x3F)
            samples.append(s)
            idx += 2

    elif adc_select == 0:
        if usb_data_type == 0:
            # 8-bit scaled USB mode from your plot code
            while idx < len(b):
                current_sample_8bit = b[idx] & 0xFF
                current_sample_float = current_sample_8bit / 150.0
                current_sample_float *= 3.3
                current_sample_int = int(current_sample_float * (2 ** adc_bits))
                samples.append(current_sample_int)
                idx += 1

        elif usb_data_type == 1:
            # 16-bit raw ADC data, big-endian exactly like old plot code
            while idx + 1 < len(b):
                s = ((b[idx] & 0xFF) << 8) | (b[idx + 1] & 0xFF)
                samples.append(s)
                idx += 2

    return np.array(samples, dtype=np.int32)


# =========================================================
# FFT DBFS
# =========================================================

def calculate_fft_dbfs(samples_u, fs, adc_bits):
    x = samples_u.astype(np.float32)

    # Convert unsigned ADC code to signed around ADC midscale
    adc_mid = 2 ** (adc_bits - 1)
    adc_fs = 2 ** (adc_bits - 1)

    x = x - adc_mid

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
    mag = mag / (len(x) * coherent_gain / 2.0)

    mag_dbfs = 20.0 * np.log10((mag / adc_fs) + 1e-15)

    return freq, mag_dbfs


def estimate_noise_floor_dbfs(mag_dbfs):
    usable = mag_dbfs[IGNORE_FIRST_BINS:]
    return np.percentile(usable, NOISE_PERCENTILE)


# =========================================================
# READ TXT FILE
# =========================================================

with open(FILENAME, "r") as f:
    lines = f.readlines()

info = parse_txt_info(lines)

fs = float(info["SAMPLING_FREQUENCY"])
samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])
adc_bits = int(info["ADC_RESOLUTION"])
hz_per_m = float(info["HZ_PER_M"])

data_lines = lines[info["DATA_START_LINE"]:]

chirps = []

for line in data_lines:
    samples = decode_usb_hex_line(line, info)

    if samples is None:
        continue

    if len(samples) >= samples_per_chirp:
        samples = samples[:samples_per_chirp]
        chirps.append(samples)

if len(chirps) == 0:
    raise RuntimeError("No valid chirps found in txt file")

num_chirps = len(chirps)

if END_CHIRP is None:
    END_CHIRP = num_chirps


# =========================================================
# PRINT INFO
# =========================================================

print("\n========== TXT LOG INFO ==========")
print(f"Record counter       : {info['RECORD_COUNTER']}")
print(f"Valid chirps         : {num_chirps}")
print(f"Record time          : {info['RECORD_TIME']} s")
print(f"Record date          : {info['RECORD_DATE']}")

print("\n----- Chirp -----")
print(f"Sweep time           : {info['SWEEP_TIME_US']} us")
print(f"Sweep gap            : {info['SWEEP_GAP_US']} us")
print(f"Sweep start          : {info['SWEEP_START'] / 1e9:.3f} GHz")
print(f"Sweep bandwidth      : {info['SWEEP_BW'] / 1e6:.1f} MHz")
print(f"Hz per meter         : {info['HZ_PER_M']} Hz/m")

max_beat_frequency = fs / 2.0
max_distance = max_beat_frequency / hz_per_m

print(f"Max beat frequency   : {max_beat_frequency / 1e6:.3f} MHz")
print(f"Max distance         : {max_distance:.2f} m")

print("\n----- Sampling -----")
print(f"Sampling frequency   : {fs / 1e6:.3f} MHz")
print(f"Samples per chirp    : {samples_per_chirp}")
print(f"ADC resolution       : {adc_bits} bit")

print("\n----- Modes -----")
print(f"TX mode              : {info['TX_MODE']}")
print(f"TX power             : {info['TX_POWER_DBM']} dBm")
print(f"TX voltage code      : {info['TX_POWER_DBM_VOLTAGE']}")
print(f"Data log             : {info['DATA_LOG']}")
print(f"ADC select           : {info['ADC_SELECT']}")
print(f"USB data type        : {info['USB_DATA_TYPE']}")
print(f"Phase distance       : {info['PHASE_DISTANCE']} cm")
print(f"CPI chirp            : {info['CPI_CHIRP']}")
print("==================================\n")


# =========================================================
# INITIAL FFT
# =========================================================

freq, mag_dbfs = calculate_fft_dbfs(chirps[START_CHIRP], fs, adc_bits)
range_m = freq / hz_per_m

noise = estimate_noise_floor_dbfs(mag_dbfs)

noise_hist = []
chirp_hist = []


# =========================================================
# PLOTS
# =========================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax_fft = axes[0]
ax_noise = axes[1]

line_fft, = ax_fft.plot(range_m, mag_dbfs)
line_noise, = ax_noise.plot([], [], label="Noise Floor")

ax_fft.set_title("USB TXT FFT dBFS/bin")
ax_noise.set_title("Noise Floor vs Chirp")

ax_fft.set_ylabel("dBFS/bin")
ax_fft.set_xlabel("Range (m)")

ax_noise.set_ylabel("Noise Floor dBFS/bin")
ax_noise.set_xlabel("Chirp Index")

ax_fft.set_xlim(0, range_m.max())
ax_fft.set_ylim(-140, 0)

ax_noise.set_ylim(-140, -40)

for ax in axes:
    ax.grid(True)

ax_noise.legend()

text_noise = ax_fft.text(
    0.02,
    0.90,
    "",
    transform=ax_fft.transAxes
)


# =========================================================
# UPDATE LOOP
# =========================================================

plt.ion()

for chirp_idx in range(START_CHIRP, END_CHIRP, CHIRP_STEP):

    samples = chirps[chirp_idx]

    freq, mag_dbfs = calculate_fft_dbfs(samples, fs, adc_bits)
    range_m = freq / hz_per_m

    noise = estimate_noise_floor_dbfs(mag_dbfs)

    chirp_hist.append(chirp_idx)
    noise_hist.append(noise)

    line_fft.set_xdata(range_m)
    line_fft.set_ydata(mag_dbfs)

    line_noise.set_data(chirp_hist, noise_hist)

    ax_noise.set_xlim(
        max(0, chirp_idx - 500),
        chirp_idx + 10
    )

    text_noise.set_text(f"Noise floor: {noise:.2f} dBFS/bin")

    fig.suptitle(f"Chirp {chirp_idx + 1}/{num_chirps}")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()