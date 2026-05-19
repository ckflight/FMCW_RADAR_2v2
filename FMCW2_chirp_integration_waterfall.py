import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/fmcw2_bin_files/10bit_dbfs_64chirp.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE  = 512
MAX_RANGE_DISPLAY = 500

NOISE_RANGE_MIN = 20
NOISE_RANGE_MAX = MAX_RANGE_DISPLAY


def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            buf[offset + 3])


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


# -----------------------------
# Read file
# -----------------------------
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File is smaller than 512-byte info sector")

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
ADC_SELECT         = info[idx]; idx += 1
USB_DATA_TYPE      = info[idx]; idx += 1
ADC_BITS           = info[idx]; idx += 1

CHIRP_END_TIMER_US      = read_u32_be(info, idx); idx += 4
CPI_END_TIMER_US        = read_u32_be(info, idx); idx += 4
CARD_WRITE_END_TIMER_US = read_u32_be(info, idx); idx += 4

CHIRPS_PER_CPI = read_u16_be(info, idx); idx += 2
CPI_COUNTER    = read_u32_be(info, idx); idx += 4


# -----------------------------
# Validate ADC bits
# -----------------------------
if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")


# -----------------------------
# Reconstruct values
# -----------------------------
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


# -----------------------------
# ADC data size
# STM32 DMA stores 10/12/14/16-bit ADC data as uint16_t
# -----------------------------
BYTES_PER_SAMPLE = 2

num_chirps = CPI_COUNTER * CHIRPS_PER_CPI
expected_samples = num_chirps * SAMPLES_PER_CHIRP
expected_bytes = expected_samples * BYTES_PER_SAMPLE

BYTES_PER_CHIRP = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE
BYTES_PER_CPI = CHIRPS_PER_CPI * BYTES_PER_CHIRP

CONFIGURED_DATA_RATE_MBPS = (BYTES_PER_CHIRP * CONFIGURED_PRF_HZ) / 1e6

CARD_WRITE_SPEED_MBPS = 0.0
if CARD_WRITE_END_TIMER_US > 0:
    CARD_WRITE_SPEED_MBPS = BYTES_PER_CPI / (CARD_WRITE_END_TIMER_US / 1e6) / 1e6


# -----------------------------
# Print info
# -----------------------------
print("\n----- SYSTEM -----")
print(f"FS                  : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
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
print(f"NUM_CHIRPS          : {num_chirps}")

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"BYTES_PER_CPI       : {BYTES_PER_CPI}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")


# -----------------------------
# Read ADC data
# -----------------------------
raw_data = file_bytes[INFO_SECTOR_SIZE : INFO_SECTOR_SIZE + expected_bytes]

data_u16 = np.frombuffer(raw_data, dtype="<u2")

if len(data_u16) < expected_samples:
    raise ValueError(
        f"Not enough ADC data: got {len(data_u16)} samples, expected {expected_samples}"
    )

data_u16 = data_u16[:expected_samples]

# Keep only valid ADC bits
ADC_MASK = (1 << ADC_BITS) - 1
data_u16 = data_u16 & ADC_MASK

print("\n----- RAW ADC CHECK -----")
print(f"Raw min             : {int(data_u16.min())}")
print(f"Raw max             : {int(data_u16.max())}")
print(f"Raw mean            : {float(data_u16.mean()):.2f}")

# Unsigned ADC -> centered signed-like float
ADC_FULL_SCALE = 1 << ADC_BITS
ADC_CENTER = 1 << (ADC_BITS - 1)

FS_PEAK = ADC_CENTER

data_adc = data_u16.astype(np.float32) - ADC_CENTER
chirps = data_adc.reshape(num_chirps, SAMPLES_PER_CHIRP)


# -----------------------------
# Range axis
# -----------------------------
freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

MAX_UNAMBIG_RANGE = np.max(range_m)
MAX_RANGE_DISPLAY = min(MAX_RANGE_DISPLAY, MAX_UNAMBIG_RANGE)

range_mask = range_m <= MAX_RANGE_DISPLAY
range_m_limited = range_m[range_mask]

print(f"MAX_UNAMBIG_RANGE   : {MAX_UNAMBIG_RANGE:.2f} m")
print(f"MAX_RANGE_DISPLAY   : {MAX_RANGE_DISPLAY:.2f} m")


# -----------------------------
# FFT normalization
# -----------------------------
w = np.hanning(SAMPLES_PER_CHIRP)
cg = np.sum(w) / SAMPLES_PER_CHIRP

# -----------------------------
# Plot setup
# -----------------------------
waterfall = np.full((CPI_COUNTER, len(range_m)), -160.0, dtype=np.float32)

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.35)

line, = ax1.plot([], [])
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Power (dBFS/bin)")
ax1.set_title("Chirp Integrated Range Profile")
ax1.grid(True)
ax1.set_xlim(0, MAX_RANGE_DISPLAY)

img = ax2.imshow(
    waterfall[:, range_mask],
    aspect="auto",
    origin="lower",
    extent=[range_m_limited[0], range_m_limited[-1], 0, CPI_COUNTER]
)
ax2.set_xlabel("Range (m)")
ax2.set_ylabel("CPI index")
ax2.set_title("Waterfall, Chirp Integrated dBFS/bin")
ax2.set_xlim(0, MAX_RANGE_DISPLAY)

rd_map_init = np.full((CHIRPS_PER_CPI, len(range_m_limited)), -160.0, dtype=np.float32)

img_rd = ax3.imshow(
    rd_map_init,
    aspect="auto",
    origin="lower",
    extent=[
        range_m_limited[0],
        range_m_limited[-1],
        -CHIRPS_PER_CPI // 2,
        CHIRPS_PER_CPI // 2,
    ],
)
ax3.set_xlabel("Range (m)")
ax3.set_ylabel("Doppler bin")
ax3.set_title("Range-Doppler Map")
ax3.set_xlim(0, MAX_RANGE_DISPLAY)


# -----------------------------
# CPI processing
# -----------------------------
for cpi_idx in range(CPI_COUNTER):

    start = cpi_idx * CHIRPS_PER_CPI
    end = start + CHIRPS_PER_CPI

    chirps_cpi = chirps[start:end]

    # Remove DC per chirp
    chirps_cpi = chirps_cpi - np.mean(chirps_cpi, axis=1, keepdims=True)

    # Window
    chirps_cpi_w = chirps_cpi * w

    # Convert ADC counts to full-scale units
    chirps_cpi_fs = chirps_cpi_w / FS_PEAK

    # Range FFT
    chirps_fft = np.fft.rfft(chirps_cpi_fs, axis=1)

    # Coherent gain / FFT amplitude normalization
    chirps_fft = chirps_fft / (SAMPLES_PER_CHIRP * cg / 2)

    # Suppress DC / close leakage bins
    chirps_fft[:, :5] = 0.0

    # Non-coherent integration over chirps
    avg_power = np.mean(np.abs(chirps_fft) ** 2, axis=0)
    avg_range_dbfs = 10 * np.log10(avg_power + 1e-30)

    # Noise floor estimate
    noise_mask = (
        (range_m > NOISE_RANGE_MIN) &
        (range_m < min(NOISE_RANGE_MAX, MAX_UNAMBIG_RANGE))
    )

    if np.any(noise_mask):
        noise_floor_dbfs = np.median(avg_range_dbfs[noise_mask])
    else:
        noise_floor_dbfs = np.median(avg_range_dbfs[10:])

    print(
        f"CPI {cpi_idx + 1:4d}/{CPI_COUNTER}: "
        f"noise floor = {noise_floor_dbfs:.2f} dBFS/bin"
    )

    waterfall[cpi_idx, :] = avg_range_dbfs

    # Range-Doppler
    doppler_fft = np.fft.fft(chirps_fft, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)
    rd_map_dbfs = 20 * np.log10(np.abs(doppler_fft) + 1e-30)
    rd_map_limited = rd_map_dbfs[:, range_mask]

    # Smooth display only
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    avg_range_dbfs_plot = np.convolve(avg_range_dbfs, kernel, mode="same")
    avg_range_limited = avg_range_dbfs_plot[range_mask]

    # Update range profile
    line.set_data(range_m_limited, avg_range_limited)

    ymax = np.max(avg_range_limited)
    ax1.set_ylim(ymax - 80, ymax + 5)

    ax1.set_title(
        f"Chirp Integrated Range Profile - CPI {cpi_idx + 1}/{CPI_COUNTER} "
        f"| Noise {noise_floor_dbfs:.2f} dBFS/bin"
    )

    # Update waterfall
    waterfall_limited = waterfall[:, range_mask]
    img.set_data(waterfall_limited)
    img.set_clim(np.max(waterfall_limited) - 40, np.max(waterfall_limited))

    # Update range-Doppler
    img_rd.set_data(rd_map_limited)
    img_rd.set_clim(np.max(rd_map_limited) - 50, np.max(rd_map_limited))

    ax3.set_title(f"Range-Doppler Map - CPI {cpi_idx + 1}/{CPI_COUNTER}")

    fig.canvas.draw_idle()
    plt.pause(0.04)

plt.ioff()
plt.show()