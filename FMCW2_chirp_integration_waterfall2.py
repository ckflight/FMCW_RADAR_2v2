import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1

USE_SYNC_HEADERS = True
HEADER_SIZE = 4

SYNC0 = 0x1C1C
SYNC1 = 0xC1C1
SYNC2 = 0x9999
SYNC3 = 0x00FF

CHIRP_STEP = 1

REMOVE_DC = True
REMOVE_FIRST_N_BINS = 0

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    BIN_FILE = "fmcw2_bin_files/road_log10_resized.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_DISPLAY = 3000


def read_u32_be(buf, offset):
    return (
        (buf[offset] << 24) |
        (buf[offset + 1] << 16) |
        (buf[offset + 2] << 8) |
        buf[offset + 3]
    )


def read_u16_be(buf, offset):
    return (buf[offset] << 8) | buf[offset + 1]


with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File is smaller than 512-byte info sector")

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
# Read ADC data with fixed stride
# -----------------------------
record_size = num_chirps_expected * BYTES_PER_CHIRP

raw_data = file_bytes[
    INFO_SECTOR_SIZE:
    INFO_SECTOR_SIZE + record_size
]

data_u16 = np.frombuffer(raw_data, dtype="<u2")

available_chirps = len(data_u16) // words_per_chirp
available_chirps = min(available_chirps, num_chirps_expected)

if available_chirps <= 0:
    raise RuntimeError("No complete chirps available")

unused_words = len(data_u16) - available_chirps * words_per_chirp

data_u16 = data_u16[:available_chirps * words_per_chirp]
chirps_raw = data_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC0) |
        (chirps_raw[:, 1] != SYNC1) |
        (chirps_raw[:, 2] != SYNC2) |
        (chirps_raw[:, 3] != SYNC3)
    )[0]

    chirps_u16 = chirps_raw[:, HEADER_SIZE:]

    print("\n----- SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps_u16)}")
    print(f"BAD HEADERS        : {len(bad_headers)}")
    print(f"UNUSED END WORDS   : {unused_words}")

    if len(bad_headers) > 0:
        print("First bad header indices:", bad_headers[:20])
else:
    chirps_u16 = chirps_raw

    print("\n----- NO SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps_u16)}")
    print(f"UNUSED END WORDS   : {unused_words}")

num_chirps = len(chirps_u16)

if num_chirps == 0:
    raise RuntimeError("No valid chirps found")

ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

adc_raw = chirps_u16 & ADC_MASK
chirps = adc_raw.astype(np.float32) - ADC_CENTER

print("\n----- RAW ADC CHECK -----")
print(f"Raw min             : {int(adc_raw.min())}")
print(f"Raw max             : {int(adc_raw.max())}")
print(f"Raw mean            : {float(adc_raw.mean()):.2f}")

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

calculated_max_range = float(range_m[-1])

PLOT_MAX_RANGE = min(MAX_RANGE_DISPLAY, calculated_max_range)

range_mask = range_m <= PLOT_MAX_RANGE
range_m_limited = range_m[range_mask]

print(f"CALCULATED MAX RANGE : {calculated_max_range:.2f} m")
print(f"PLOT MAX RANGE       : {PLOT_MAX_RANGE:.2f} m")

FULL_CPI_COUNT = num_chirps // CHIRPS_PER_CPI

if FULL_CPI_COUNT == 0:
    raise ValueError("Not enough chirps for one full CPI")

print(f"FULL CPI COUNT     : {FULL_CPI_COUNT}")

DISPLAY_CPI_COUNT = (FULL_CPI_COUNT + CHIRP_STEP - 1) // CHIRP_STEP

waterfall = np.zeros((DISPLAY_CPI_COUNT, len(range_m)), dtype=np.float32)

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.35)

line, = ax1.plot([], [])
ax1.set_xlabel("Range")
ax1.set_ylabel("Power")
ax1.set_title("Range Profile (CPI)")
ax1.grid(True)
ax1.set_xlim(0, PLOT_MAX_RANGE)

img = ax2.imshow(
    np.zeros_like(waterfall[:, range_mask]),
    aspect="auto",
    origin="lower",
    extent=[range_m_limited[0], range_m_limited[-1], 0, DISPLAY_CPI_COUNT]
)
ax2.set_xlabel("Range")
ax2.set_ylabel("CPI index")
ax2.set_title("Waterfall")
ax2.set_xlim(0, PLOT_MAX_RANGE)

rd_map_init = np.zeros((CHIRPS_PER_CPI, len(range_m_limited)), dtype=np.float32)

img_rd = ax3.imshow(
    rd_map_init,
    aspect="auto",
    origin="lower",
    extent=[
        range_m_limited[0],
        range_m_limited[-1],
        -CHIRPS_PER_CPI // 2,
        CHIRPS_PER_CPI // 2,
    ]
)
ax3.set_xlabel("Range")
ax3.set_ylabel("Doppler bin")
ax3.set_title("Range-Doppler")
ax3.set_xlim(0, PLOT_MAX_RANGE)

for cpi_idx in range(0, FULL_CPI_COUNT, CHIRP_STEP):

    start = cpi_idx * CHIRPS_PER_CPI
    end = start + CHIRPS_PER_CPI

    chirps_cpi = chirps[start:end]

    if REMOVE_DC:
        chirps_cpi = chirps_cpi - np.mean(chirps_cpi, axis=1, keepdims=True)

    chirps_fft = np.fft.rfft(chirps_cpi, axis=1)

    chirps_fft[:, :REMOVE_FIRST_N_BINS] = 0

    doppler_fft = np.fft.fft(chirps_fft, axis=0)
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)

    rd_map = 20 * np.log10(np.abs(doppler_fft) + 1e-12)
    rd_map_limited = rd_map[:, range_mask]

    avg_range = np.mean(np.abs(chirps_fft) ** 2, axis=0)

    display_idx = cpi_idx // CHIRP_STEP
    waterfall[display_idx, :] = avg_range

    avg_range_limited = avg_range[range_mask]

    line.set_data(range_m_limited, avg_range_limited)
    ax1.set_ylim(np.min(avg_range_limited), np.max(avg_range_limited) * 1.1)
    ax1.set_title(f"Range Profile - CPI {cpi_idx + 1}/{FULL_CPI_COUNT}")

    waterfall_db = 10 * np.log10(waterfall[:, range_mask] + 1e-12)
    img.set_data(waterfall_db)
    img.set_clim(np.max(waterfall_db) - 25, np.max(waterfall_db))

    img_rd.set_data(rd_map_limited)
    img_rd.set_clim(np.max(rd_map_limited) - 20, np.max(rd_map_limited))
    ax3.set_title(f"Range-Doppler - CPI {cpi_idx + 1}/{FULL_CPI_COUNT}")

    fig.canvas.draw_idle()
    plt.pause(0.01)

plt.ioff()
plt.show()