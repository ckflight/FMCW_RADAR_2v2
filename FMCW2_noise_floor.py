import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# This code works with both log files with snyc header or no sync header.
# This code is for .bin files logged with radar2v2 sdcard

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

USE_SYNC_HEADERS = True   # True = old sync logs, False = current no-sync logs
SYNC = 0xC8C8

if OPERATING_SYSTEM == 1:
    #BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    BIN_FILE = "fmcw2_bin_files/salon_run_att.bin"
    BIN_FILE = "fmcw2_bin_files/10bit_64_sync_salon_run_tx3db_rx6db.bin"
    #BIN_FILE = "Radar_Records/data_record.bin"

elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

START_CHIRP = 0
END_CHIRP = None
CHIRP_STEP = 100
FRAME_DELAY = 0.001

REMOVE_DC = True
USE_WINDOW = True

IGNORE_FIRST_BINS = 0
NOISE_PERCENTILE = 20

INFO_SECTOR_SIZE = 512

BIN_SAMPLE_DTYPE = "<u2"
BIN_CHIRP_HEADER_SAMPLES = 0

# =========================================================
# TXT INFO PARSER
# =========================================================

def read_int_line(lines, idx):
    return int(lines[idx].strip()), idx + 1


def parse_txt_info(lines):
    p = {}
    idx = 0

    names = [
        "RECORD_COUNTER",
        "RECORD_TIME",
        "SWEEP_TIME_US",
        "SWEEP_GAP_US",
        "SWEEP_START",
        "SWEEP_BW",
        "SAMPLING_FREQUENCY",
        "NUMBER_OF_SAMPLES",
        "TX_MODE",
        "TX_POWER_DBM",
        "TX_POWER_DBM_VOLTAGE",
        "HZ_PER_M",
        "DATA_LOG",
        "USB_DATA_TYPE",
        "ADC_RESOLUTION",
        "PHASE_DISTANCE",
    ]

    for name in names:
        p[name], idx = read_int_line(lines, idx)

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
# BIN INFO PARSER
# =========================================================

def u32_be(buf, off):
    return (
        (buf[off] << 24)
        | (buf[off + 1] << 16)
        | (buf[off + 2] << 8)
        | buf[off + 3]
    )


def parse_bin_info(raw512):
    p = {}
    idx = 0

    p["RECORD_COUNTER"] = u32_be(raw512, idx); idx += 4
    p["RECORD_TIME"] = u32_be(raw512, idx); idx += 4
    p["SWEEP_TIME_US"] = u32_be(raw512, idx); idx += 4
    p["SWEEP_GAP_US"] = u32_be(raw512, idx); idx += 4

    p["SWEEP_START"] = u32_be(raw512, idx) * 1e7; idx += 4
    p["SWEEP_BW"] = u32_be(raw512, idx) * 1e6; idx += 4
    p["SAMPLING_FREQUENCY"] = u32_be(raw512, idx) * 1000; idx += 4

    p["NUMBER_OF_SAMPLES"] = u32_be(raw512, idx); idx += 4

    p["TX_MODE"] = raw512[idx]; idx += 1
    p["TX_POWER_DBM"] = raw512[idx]; idx += 1
    p["TX_POWER_DBM_VOLTAGE"] = raw512[idx]; idx += 1

    p["HZ_PER_M"] = u32_be(raw512, idx); idx += 4

    p["DATA_LOG"] = raw512[idx]; idx += 1
    p["USB_DATA_TYPE"] = raw512[idx]; idx += 1
    p["ADC_RESOLUTION"] = raw512[idx]; idx += 1

    p["CHIRP_END_TIMER_US"] = u32_be(raw512, idx); idx += 4
    p["CPI_END_TIMER_US"] = u32_be(raw512, idx); idx += 4
    p["CARD_WRITE_END_TIMER_US"] = u32_be(raw512, idx); idx += 4

    p["CPI_CHIRP"] = (raw512[idx] << 8) | raw512[idx + 1]; idx += 2
    p["CPI_COUNTER"] = u32_be(raw512, idx); idx += 4

    p["PHASE_DISTANCE"] = 0

    date_bytes = raw512[64:128].split(b"\x00")[0]
    p["RECORD_DATE"] = date_bytes.decode("ascii", errors="ignore").strip()

    return p


# =========================================================
# USB TXT HEX DECODE
# =========================================================

def decode_usb_hex_line(hex_line, info):
    hex_line = hex_line.strip()

    if len(hex_line) == 0:
        return None

    b = bytes.fromhex(hex_line)

    usb_data_type = int(info["USB_DATA_TYPE"])
    adc_bits = int(info["ADC_RESOLUTION"])

    samples = []
    idx = 0

    if usb_data_type == 0:
        while idx < len(b):
            current_sample_8bit = b[idx] & 0xFF
            current_sample_float = current_sample_8bit / 150.0
            current_sample_float *= 3.3
            current_sample_int = int(current_sample_float * (2 ** adc_bits))
            samples.append(current_sample_int)
            idx += 1

    elif usb_data_type == 1:
        while idx + 1 < len(b):
            s = ((b[idx] & 0xFF) << 8) | (b[idx + 1] & 0xFF)
            samples.append(s)
            idx += 2

    return np.array(samples, dtype=np.int32)


# =========================================================
# LOADERS
# =========================================================

def load_txt_record(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    info = parse_txt_info(lines)
    samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])

    chirps = []

    for line in lines[info["DATA_START_LINE"]:]:
        samples = decode_usb_hex_line(line, info)

        if samples is None:
            continue

        if len(samples) >= samples_per_chirp:
            chirps.append(samples[:samples_per_chirp])

    if len(chirps) == 0:
        raise RuntimeError("No valid chirps found in txt file")

    return info, np.array(chirps, dtype=np.int32), "USB TXT"


def extract_chirps_with_sync(data, samples_per_chirp):
    sync_idx = np.where(
        (data[:-1] == SYNC) &
        (data[1:] == SYNC)
    )[0]

    chirps = []

    for idx in sync_idx:
        start = idx + 2
        end = start + samples_per_chirp

        if end <= len(data):
            chirp = data[start:end]

            if len(chirp) == samples_per_chirp:
                chirps.append(chirp)

    chirps = np.array(chirps, dtype=np.int32)

    print("\n----- SYNC MODE -----")
    print(f"Sync word            : 0x{SYNC:04X}")
    print(f"Found sync headers   : {len(sync_idx)}")
    print(f"Valid synced chirps  : {len(chirps)}")

    return chirps


def extract_chirps_without_sync(data, samples_per_chirp):
    total_samples_per_chirp = samples_per_chirp + BIN_CHIRP_HEADER_SAMPLES

    usable_samples = (len(data) // total_samples_per_chirp) * total_samples_per_chirp
    unused_samples = len(data) - usable_samples

    data = data[:usable_samples]

    chirps_raw = data.reshape(-1, total_samples_per_chirp)

    if BIN_CHIRP_HEADER_SAMPLES > 0:
        chirps = chirps_raw[:, BIN_CHIRP_HEADER_SAMPLES:]
    else:
        chirps = chirps_raw

    print("\n----- NO SYNC MODE -----")
    print(f"Continuous chirps    : {len(chirps)}")
    print(f"Unused end samples   : {unused_samples}")

    return chirps.astype(np.int32)


def load_bin_record(filename):
    with open(filename, "rb") as f:
        raw = f.read()

    if len(raw) <= INFO_SECTOR_SIZE:
        raise RuntimeError("BIN file is too small")

    info = parse_bin_info(raw[:INFO_SECTOR_SIZE])

    samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])

    data = np.frombuffer(
        raw[INFO_SECTOR_SIZE:],
        dtype=np.dtype(BIN_SAMPLE_DTYPE)
    ).astype(np.int32)

    if USE_SYNC_HEADERS:
        chirps = extract_chirps_with_sync(data, samples_per_chirp)
        source_type = "SD BIN SYNC"
    else:
        chirps = extract_chirps_without_sync(data, samples_per_chirp)
        source_type = "SD BIN NO SYNC"

    if len(chirps) == 0:
        raise RuntimeError("No valid chirps found in bin file")

    return info, chirps, source_type


def load_record(filename):
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        return load_txt_record(filename)

    if suffix == ".bin":
        return load_bin_record(filename)

    raise RuntimeError("Unsupported file type. Use .txt or .bin")


# =========================================================
# FFT DBFS
# =========================================================

def calculate_fft_dbfs(samples_u, fs, adc_bits):
    adc_mid = 2 ** (adc_bits - 1)
    adc_fs = 2 ** (adc_bits - 1)

    adc_mask = (1 << adc_bits) - 1

    x = samples_u.astype(np.int32)
    x = x & adc_mask

    x = x.astype(np.float32)
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
# LOAD FILE
# =========================================================

info, chirps, source_type = load_record(BIN_FILE)

fs = float(info["SAMPLING_FREQUENCY"])
samples_per_chirp = int(info["NUMBER_OF_SAMPLES"])
adc_bits = int(info["ADC_RESOLUTION"])
hz_per_m = float(info["HZ_PER_M"])

num_chirps = len(chirps)

if END_CHIRP is None:
    END_CHIRP = num_chirps

END_CHIRP = min(END_CHIRP, num_chirps)

# =========================================================
# PRINT INFO
# =========================================================

print(f"\n========== {source_type} LOG INFO ==========")
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
print(f"USB data type        : {info['USB_DATA_TYPE']}")
print(f"Phase distance       : {info['PHASE_DISTANCE']} cm")
print(f"CPI chirp            : {info['CPI_CHIRP']}")
print("==================================\n")

# =========================================================
# INITIAL FFT
# =========================================================

freq, mag_dbfs = calculate_fft_dbfs(chirps[START_CHIRP], fs, adc_bits)
range_m = freq / hz_per_m

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

ax_fft.set_title(f"{source_type} FFT dBFS/bin")
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

    fig.suptitle(f"{source_type} - Chirp {chirp_idx + 1}/{num_chirps}")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()