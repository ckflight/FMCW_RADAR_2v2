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

CHIRP_STEP = 64
FRAME_DELAY = 0.0001

REMOVE_DC = True
USE_WINDOW = True
IGNORE_FIRST_BINS = 10
NOISE_PERCENTILE = 20

INFO_SECTOR_SIZE = 512

if OPERATING_SYSTEM == 1:
    BIN_FILE = "Radar_Records/data_record.bin"
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    BIN_FILE = "fmcw2_bin_files/sar_log4.bin"
else:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"


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
print(f"NUM_CHIRPS          : {expected_chirps}")

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"BYTES_PER_CPI       : {BYTES_PER_CPI}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")


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
# ADC handling
# -----------------------------
ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = 1 << (ADC_BITS - 1)
ADC_FS = ADC_CENTER

chirps = chirps & ADC_MASK

print("\n----- RAW ADC CHECK -----")
print(f"Raw min            : {int(chirps.min())}")
print(f"Raw max            : {int(chirps.max())}")
print(f"Raw mean           : {float(chirps.mean()):.2f}")

chirps = chirps.astype(np.float32) - ADC_CENTER

# Normalize data
chirps = chirps / ADC_FS

# -----------------------------
# FFT function
# -----------------------------
def calculate_fft_dbfs(samples):
    x = samples.astype(np.float32)

    if REMOVE_DC:
        x = x - np.mean(x)

    if USE_WINDOW:
        window = np.hanning(len(x))
        coherent_gain = np.sum(window) / len(window)
        x = x * window
    else:
        coherent_gain = 1.0

    
    fft_data = np.fft.rfft(x)

    fft_data_normalized = fft_data / (SAMPLES_PER_CHIRP * coherent_gain / 2)
    freq = np.fft.rfftfreq(len(x), d=1.0 / FS)

    mag = np.abs(fft_data_normalized)
    
    # Adc data is normalized to 1 -1 range so reference is 0dbfs 
    # therefore, 20log conversion gives dbfs
    mag_dbfs = 20.0 * np.log10(mag + 1e-15)

    return freq, mag_dbfs


def estimate_noise_floor_dbfs(mag_dbfs):
    usable = mag_dbfs[IGNORE_FIRST_BINS:]
    return np.percentile(usable, NOISE_PERCENTILE)


# -----------------------------
# Initial FFT
# -----------------------------
freq, mag_dbfs = calculate_fft_dbfs(chirps[0])

if HZ_PER_M > 0:
    range_m = freq / HZ_PER_M
else:
    range_m = freq

noise_hist = []
chirp_hist = []


# -----------------------------
# Plot setup
# -----------------------------
plt.ion()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax_fft = axes[0]
ax_noise = axes[1]

line_fft, = ax_fft.plot(range_m, mag_dbfs)
line_noise, = ax_noise.plot([], [], label="Noise Floor")

ax_fft.set_title("SD BIN SYNC FFT dBFS/bin")
ax_fft.set_xlabel("Range (m)")
ax_fft.set_ylabel("dBFS/bin")
ax_fft.set_xlim(0, range_m.max())
ax_fft.set_ylim(-140, 0)
ax_fft.grid(True)

ax_noise.set_title("Noise Floor vs Chirp")
ax_noise.set_xlabel("Chirp Index")
ax_noise.set_ylabel("Noise Floor dBFS/bin")
ax_noise.set_ylim(-140, -40)
ax_noise.grid(True)
ax_noise.legend()

text_noise = ax_fft.text(
    0.02,
    0.90,
    "",
    transform=ax_fft.transAxes
)


# -----------------------------
# Plot only loaded/header-limited chirps
# -----------------------------
for chirp_idx in range(0, num_chirps, CHIRP_STEP):

    freq, mag_dbfs = calculate_fft_dbfs(chirps[chirp_idx])

    if HZ_PER_M > 0:
        range_m = freq / HZ_PER_M
    else:
        range_m = freq

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

    fig.suptitle(f"SD BIN SYNC - Chirp {chirp_idx + 1}/{num_chirps}")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()