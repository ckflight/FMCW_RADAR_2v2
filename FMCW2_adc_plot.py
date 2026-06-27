import numpy as np
import matplotlib.pyplot as plt

# This code is for .bin files logged with radar2v2 sdcard

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

USE_SYNC_HEADERS = True
SYNC = 0xC8C8

CHIRP_STEP = 1   # 1 = every chirp, 2 = every 2nd chirp, 4 = every 4th chirp

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
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
expected_chirps = CPI_COUNTER * CHIRPS_PER_CPI

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

if expected_chirps <= 0:
    raise ValueError("expected_chirps is zero")


# -----------------------------
# Print info
# -----------------------------
print("\n----- INFO -----")
print(f"FS                : {FS / 1e6:.3f} MHz")
print(f"ADC_BITS          : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS   : {expected_chirps}")


# -----------------------------
# Read only current recording according to CPI_COUNTER
# -----------------------------
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + 2   # 2 sync words + ADC samples
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_to_read = expected_chirps * words_per_chirp * 2

raw_data = file_bytes[
    INFO_SECTOR_SIZE :
    INFO_SECTOR_SIZE + bytes_to_read
]

adc_u16 = np.frombuffer(raw_data, dtype="<u2")

available_chirps = len(adc_u16) // words_per_chirp
available_chirps = min(available_chirps, expected_chirps)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps found")

adc_u16 = adc_u16[:available_chirps * words_per_chirp]
chirps_raw = adc_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC) |
        (chirps_raw[:, 1] != SYNC)
    )[0]

    if len(bad_headers) > 0:
        print(f"WARNING: {len(bad_headers)} bad sync headers")

    chirps = chirps_raw[:, 2:]
else:
    chirps = chirps_raw

num_chirps = len(chirps)

print(f"LOADED CHIRPS      : {num_chirps}")
print(f"LOADED CPI         : {num_chirps // CHIRPS_PER_CPI}")


# -----------------------------
# Generic ADC bit handling
# -----------------------------
ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

adc_raw = chirps & ADC_MASK
adc_centered = adc_raw.astype(np.float32) - ADC_CENTER

eps = 1e-12
adc_norm = adc_centered / ADC_CENTER
adc_dbfs = 20.0 * np.log10(np.abs(adc_norm) + eps)

print("\n----- RAW ADC CHECK -----")
print(f"Raw min       : {adc_raw.min()}")
print(f"Raw max       : {adc_raw.max()}")
print(f"Centered min  : {adc_centered.min():.1f}")
print(f"Centered max  : {adc_centered.max():.1f}")
print(f"Centered mean : {adc_centered.mean():.2f}")
print(f"Peak dBFS     : {adc_dbfs.max():.2f} dBFS")


# -----------------------------
# Plot each chirp ADC waveform
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))

for chirp_idx in range(0, num_chirps, CHIRP_STEP):

    chirp_centered = adc_centered[chirp_idx]

    peak = np.max(np.abs(chirp_centered)) / ADC_CENTER
    peak_dbfs = 20.0 * np.log10(peak + eps)

    cpi_idx = chirp_idx // CHIRPS_PER_CPI
    chirp_in_cpi = chirp_idx % CHIRPS_PER_CPI

    ax.clear()
    ax.plot(chirp_centered)

    ax.set_title(
        f"Raw ADC Chirp {chirp_idx + 1}/{num_chirps} | "
        f"CPI {cpi_idx + 1}/{CPI_COUNTER}, Chirp {chirp_in_cpi + 1}/{CHIRPS_PER_CPI} | "
        f"Peak = {peak_dbfs:.2f} dBFS"
    )
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("ADC value centered")
    ax.grid(True)

    plt.pause(0.02)

plt.ioff()
plt.show()