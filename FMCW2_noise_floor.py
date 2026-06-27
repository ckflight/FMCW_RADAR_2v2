import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1

USE_SYNC_HEADERS = True
SYNC = 0xC8C8

CHIRP_STEP = 10
FRAME_DELAY = 0.0001

REMOVE_DC = False
USE_WINDOW = True
IGNORE_FIRST_BINS = 0
NOISE_PERCENTILE = 20

INFO_SECTOR_SIZE = 512

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
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

# -----------------------------
# Decode info sector
# -----------------------------
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
# Basic values
# -----------------------------
FS = FS_KHZ * 1000
num_chirps_expected = CPI_COUNTER * CHIRPS_PER_CPI

if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if num_chirps_expected <= 0:
    raise RuntimeError("CPI_COUNTER or CHIRPS_PER_CPI is zero")

print("\n----- INFO -----")
print(f"FS                : {FS / 1e6:.3f} MHz")
print(f"ADC_BITS          : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP : {SAMPLES_PER_CHIRP}")
print(f"CHIRPS_PER_CPI    : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER       : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS   : {num_chirps_expected}")

# -----------------------------
# Read only CPI_COUNTER amount of ADC data
# -----------------------------
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + 2
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_per_chirp = words_per_chirp * 2
bytes_to_read = num_chirps_expected * bytes_per_chirp

raw_data = file_bytes[
    INFO_SECTOR_SIZE :
    INFO_SECTOR_SIZE + bytes_to_read
]

data_u16 = np.frombuffer(raw_data, dtype="<u2")

available_chirps = len(data_u16) // words_per_chirp
available_chirps = min(available_chirps, num_chirps_expected)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps available")

data_u16 = data_u16[:available_chirps * words_per_chirp]
chirps_raw = data_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC) |
        (chirps_raw[:, 1] != SYNC)
    )[0]

    if len(bad_headers) > 0:
        print(f"WARNING: {len(bad_headers)} chirp headers are not 0xC8C8")

    chirps = chirps_raw[:, 2:]
else:
    chirps = chirps_raw

num_chirps = len(chirps)

print(f"LOADED CHIRPS      : {num_chirps}")
print(f"LOADED CPI         : {num_chirps // CHIRPS_PER_CPI}")

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

# -----------------------------
# FFT function
# -----------------------------
def calculate_fft_dbfs(samples_u):
    x = samples_u.astype(np.int32)
    x = x & ADC_MASK
    x = x.astype(np.float32) - ADC_CENTER

    if REMOVE_DC:
        x = x - np.mean(x)

    if USE_WINDOW:
        window = np.hanning(len(x))
        coherent_gain = np.sum(window) / len(window)
        x = x * window
    else:
        coherent_gain = 1.0

    fft_data = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x), d=1.0 / FS)

    mag = np.abs(fft_data)
    mag = mag / (len(x) * coherent_gain / 2.0)

    mag_dbfs = 20.0 * np.log10((mag / ADC_FS) + 1e-15)

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