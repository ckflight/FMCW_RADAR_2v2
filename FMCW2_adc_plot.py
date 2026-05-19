import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/fmcw2_bin_files/16bit_dbfs_128chirp.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512


def read_u32_be(buf, offset):
    return (
        (buf[offset] << 24) |
        (buf[offset + 1] << 16) |
        (buf[offset + 2] << 8) |
        buf[offset + 3]
    )


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


# -----------------------------
# Read file
# -----------------------------
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File is smaller than info sector")


# -----------------------------
# Decode info sector
# -----------------------------
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
ADC_SELECT         = info[idx]; idx += 1
USB_DATA_TYPE      = info[idx]; idx += 1
ADC_BITS           = info[idx]; idx += 1

CHIRP_END_TIMER_US      = read_u32_be(info, idx); idx += 4
CPI_END_TIMER_US        = read_u32_be(info, idx); idx += 4
CARD_WRITE_END_TIMER_US = read_u32_be(info, idx); idx += 4

CHIRPS_PER_CPI = read_u16_be(info, idx); idx += 2
CPI_COUNTER    = read_u32_be(info, idx); idx += 4


# -----------------------------
# Validate parameters
# -----------------------------
if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

FS = FS_KHZ * 1000
num_chirps = CPI_COUNTER * CHIRPS_PER_CPI

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

if num_chirps <= 0:
    raise ValueError("num_chirps is zero")


# -----------------------------
# Print info
# -----------------------------
print("\n----- INFO -----")
print(f"FS                : {FS / 1e6:.3f} MHz")
print(f"ADC_BITS          : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")
print(f"NUM_CHIRPS        : {num_chirps}")


# -----------------------------
# Read ADC samples
# STM32 ADC DMA stores 10/12/14/16-bit samples inside uint16_t
# File data is little-endian uint16_t
# -----------------------------
expected_samples = num_chirps * SAMPLES_PER_CHIRP
expected_bytes = expected_samples * 2

raw_data = file_bytes[INFO_SECTOR_SIZE : INFO_SECTOR_SIZE + expected_bytes]

if len(raw_data) < expected_bytes:
    raise ValueError(
        f"Not enough raw bytes: got {len(raw_data)}, expected {expected_bytes}"
    )

adc_u16 = np.frombuffer(raw_data, dtype="<u2", count=expected_samples)


# -----------------------------
# Generic ADC bit handling
# Works for 10/12/14/16-bit unsigned ADC samples
# -----------------------------
ADC_MASK = (1 << ADC_BITS) - 1
ADC_FULL_SCALE = float(1 << ADC_BITS)
ADC_CENTER = float(1 << (ADC_BITS - 1))

adc_raw = adc_u16 & ADC_MASK
adc_centered = adc_raw.astype(np.float32) - ADC_CENTER

# Normalized ADC value, approximately -1.0 to +1.0
adc_norm = adc_centered / ADC_CENTER

# dBFS sample amplitude
eps = 1e-12
adc_dbfs = 20.0 * np.log10(np.abs(adc_norm) + eps)


print("\n----- RAW ADC CHECK -----")
print(f"Raw min       : {adc_raw.min()}")
print(f"Raw max       : {adc_raw.max()}")
print(f"Centered min  : {adc_centered.min():.1f}")
print(f"Centered max  : {adc_centered.max():.1f}")
print(f"Centered mean : {adc_centered.mean():.2f}")
print(f"Peak dBFS     : {adc_dbfs.max():.2f} dBFS")


# -----------------------------
# Reshape into chirps
# -----------------------------
chirps = adc_centered.reshape(num_chirps, SAMPLES_PER_CHIRP)
chirps_norm = adc_norm.reshape(num_chirps, SAMPLES_PER_CHIRP)


# -----------------------------
# Plot average raw ADC per CPI
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

for cpi_idx in range(CPI_COUNTER):
    start = cpi_idx * CHIRPS_PER_CPI
    end = start + CHIRPS_PER_CPI

    chirps_cpi = chirps[start:end]
    avg_chirp = np.mean(chirps_cpi, axis=0)

    peak = np.max(np.abs(avg_chirp)) / ADC_CENTER
    peak_dbfs = 20.0 * np.log10(peak + eps)

    ax.clear()
    ax.plot(avg_chirp)

    ax.set_title(
        f"Average Raw ADC CPI {cpi_idx + 1}/{CPI_COUNTER} | "
        f"Peak = {peak_dbfs:.2f} dBFS"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("ADC value centered")
    ax.grid(True)

    plt.pause(0.05)

plt.ioff()
plt.show()