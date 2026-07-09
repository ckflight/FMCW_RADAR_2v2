import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

OPERATING_SYSTEM = 1

USE_SYNC_HEADERS = True
HEADER_SIZE = 4

SYNC0 = 0x1C1C
SYNC1 = 0xC1C1
SYNC2 = 0x9999
SYNC3 = 0x00FF

CHIRP_STEP = 8

if OPERATING_SYSTEM == 1:
    BIN_FILE = "Radar_Records/data_record.bin"
    #BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    BIN_FILE = "fmcw2_bin_files/sar_log4.bin"

else:
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


if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

num_chirps_expected = CPI_COUNTER * CHIRPS_PER_CPI

if num_chirps_expected <= 0:
    raise ValueError("num_chirps_expected is zero")


FS = FS_KHZ * 1000
SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6

CONFIGURED_PRF_HZ = 0.0
if (SWEEP_TIME_US + SWEEP_GAP_US) > 0:
    CONFIGURED_PRF_HZ = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

MEASURED_CHIRP_RATE_HZ = 0.0
if CHIRP_END_TIMER_US > 0:
    MEASURED_CHIRP_RATE_HZ = 1e6 / CHIRP_END_TIMER_US

CPI_RATE_HZ = 0.0
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    CPI_RATE_HZ = 1e6 / (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US)


BYTES_PER_SAMPLE = 2

if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

BYTES_PER_CHIRP = words_per_chirp * BYTES_PER_SAMPLE
BYTES_PER_CPI = CHIRPS_PER_CPI * BYTES_PER_CHIRP

CONFIGURED_DATA_RATE_MBPS = (BYTES_PER_CHIRP * CONFIGURED_PRF_HZ) / 1e6

CARD_WRITE_SPEED_MBPS = 0.0
if CARD_WRITE_END_TIMER_US > 0:
    CARD_WRITE_SPEED_MBPS = BYTES_PER_CPI / (CARD_WRITE_END_TIMER_US / 1e6) / 1e6


print("\n----- SYSTEM -----")
print(f"FS                  : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"HEADER_SIZE         : {HEADER_SIZE if USE_SYNC_HEADERS else 0} words")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"ADC_BITS            : {ADC_BITS}")
print(f"TX_POWER            : {TX_POWER_DBM} dBm")
print(f"TX_VOLT             : {TX_POWER_DBM_VOLT}")
print(f"SWEEP_START         : {SWEEP_START/1e6:.2f} MHz")
print(f"SWEEP_BW            : {SWEEP_BW/1e6:.2f} MHz")

print("\n----- TIMING -----")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")

print("\n----- CPI -----")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_RATE            : {CPI_RATE_HZ:.2f} Hz")
print(f"CPI_COUNTER         : {CPI_COUNTER}")
print(f"NUM_CHIRPS          : {num_chirps_expected}")

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"BYTES_PER_CPI       : {BYTES_PER_CPI}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")

# -----------------------------
# Validate
# -----------------------------
if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")

expected_chirps = CPI_COUNTER * CHIRPS_PER_CPI

if expected_chirps <= 0:
    raise ValueError("expected_chirps is zero")


# -----------------------------
# Print info
# -----------------------------
print("\n----- INFO -----")
print(f"FS                : {FS_KHZ / 1000:.3f} MHz")
print(f"ADC_BITS          : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS   : {expected_chirps}")


# -----------------------------
# Read only current record
# -----------------------------
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_to_read = expected_chirps * words_per_chirp * 2
print(f"Bytes to read: {bytes_to_read}")

raw_data = file_bytes[
    INFO_SECTOR_SIZE :
    INFO_SECTOR_SIZE + bytes_to_read
]
print(f"Raw data length: {len(raw_data)}")

adc_u16 = np.frombuffer(raw_data, dtype="<u2")
print(f"Raw data u16 length: {len(adc_u16)}")

# -----------------------------
# Extract chirps with fixed stride
# -----------------------------
available_chirps = len(adc_u16) // words_per_chirp
available_chirps = min(available_chirps, expected_chirps)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps available")

unused_words = len(adc_u16) - available_chirps * words_per_chirp

adc_u16 = adc_u16[:available_chirps * words_per_chirp]

# 2d array is created with number of chirps, number of words per one chirp (number of 16 bit adc data)
chirps_raw = adc_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC0) |
        (chirps_raw[:, 1] != SYNC1) |
        (chirps_raw[:, 2] != SYNC2) |
        (chirps_raw[:, 3] != SYNC3)
    )[0]

    # take each row and to each row get rid of header from adc data
    chirps = chirps_raw[:, HEADER_SIZE:]

    print("\n----- SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"BAD HEADERS        : {len(bad_headers)}")
    print(f"UNUSED END WORDS   : {unused_words}")
    print(f"VALID CPI          : {len(chirps) // CHIRPS_PER_CPI}")

    if len(bad_headers) > 0:
        print("First bad header indices:", bad_headers[:20])

else:
    chirps = chirps_raw

    print("\n----- NO SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"UNUSED END WORDS   : {unused_words}")
    print(f"VALID CPI          : {len(chirps) // CHIRPS_PER_CPI}")

num_chirps = len(chirps)

if num_chirps == 0:
    raise RuntimeError("No valid chirps found")


# -----------------------------
# ADC conversion
# -----------------------------
ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

# adc_raw is 2d array as well
adc_raw = chirps & ADC_MASK

# convert integer data to float and substract 2^11 to center the data around 0
adc_centered = adc_raw.astype(np.float32) - ADC_CENTER

# Normalization makes reference full scale as 0 dBfs
# eps is to prevent log(0) error.
eps = 1e-12
adc_norm = adc_centered / ADC_CENTER
adc_dbfs = 20.0 * np.log10(np.abs(adc_norm) + eps)

print("\n----- ADC CHECK -----")
print(f"Raw min       : {adc_raw.min()}")
print(f"Raw max       : {adc_raw.max()}")
print(f"Centered mean : {adc_centered.mean():.2f}")
print(f"Peak dBFS     : {adc_dbfs.max():.2f} dBFS")


#-----------------------------
# Window (to see the affect on time domain data before fft)
#-----------------------------
window = np.hanning(SAMPLES_PER_CHIRP)

# -----------------------------
# Plot each chirp
# -----------------------------
plt.ion() # interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

for chirp_idx in range(0, num_chirps, CHIRP_STEP):

    # adc_center is 2d data, select entire row whic is each chirp's data.
    y = adc_centered[chirp_idx]
    y_windowed = y * window

    peak = np.max(np.abs(y)) / ADC_CENTER
    peak_dbfs = 20.0 * np.log10(peak + eps)

    cpi_idx = chirp_idx // CHIRPS_PER_CPI
    chirp_in_cpi = chirp_idx % CHIRPS_PER_CPI

    ax.clear()
    ax.plot(y, label = "Raw", alpha=0.6)
    ax.plot(y_windowed, label = "Windowed (Hanning)", linewidth=2)
    ax.legend(loc="upper right")
    ax.set_ylim(ymin=-1024,ymax=1024)

    ax.set_title(
        f"ADC Chirp {chirp_idx + 1}/{num_chirps} | "
        f"CPI {cpi_idx + 1}/{CPI_COUNTER} | "
        f"Chirp {chirp_in_cpi + 1}/{CHIRPS_PER_CPI} | "
        f"Peak {peak_dbfs:.2f} dBFS"
    )

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("ADC centered")
    ax.grid(True)

    plt.pause(0.001)

plt.ioff()
plt.show()