import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

# This code is for .bin files logged with radar2v2 sdcard

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

USE_SYNC_HEADERS = True
HEADER_SIZE = 4

SYNC0 = 0x1C1C
SYNC1 = 0xC1C1
SYNC2 = 0x9999
SYNC3 = 0x00FF

CHIRP_STEP = 1

REMOVE_DC = True
REMOVE_FIRST_N_BINS = 0

# Set noise floor threshold, higher value more detail closer to noise floor
WATERFALL_NF_DB = 40
DOPPLER_NF_DB = 45

if OPERATING_SYSTEM == 1:
    #BIN_FILE = "Radar_Records/data_record.bin"
    #BIN_FILE = "/home/ck/Desktop/horn_200mhz_5G95_1000us_32_cleanplot.bin"
    BIN_FILE = "fmcw2_bin_files/road_log6_resized.bin"

elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE  = 512
MAX_RANGE_DISPLAY = 50000

NOISE_RANGE_MIN = 0
NOISE_RANGE_MAX = MAX_RANGE_DISPLAY



def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            buf[offset + 3])


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
chirps_raw = adc_u16.reshape(available_chirps, words_per_chirp)

if USE_SYNC_HEADERS:
    bad_headers = np.where(
        (chirps_raw[:, 0] != SYNC0) |
        (chirps_raw[:, 1] != SYNC1) |
        (chirps_raw[:, 2] != SYNC2) |
        (chirps_raw[:, 3] != SYNC3)
    )[0]

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


FULL_CPI_COUNT = num_chirps // CHIRPS_PER_CPI
DISPLAY_CPI_COUNT = FULL_CPI_COUNT // CHIRP_STEP

if FULL_CPI_COUNT == 0:
    raise RuntimeError("Not enough chirps for one full CPI")

print(f"FULL CPI COUNT     : {FULL_CPI_COUNT}")


ADC_MASK = (1 << ADC_BITS) - 1
chirps = chirps & ADC_MASK

print("\n----- RAW ADC CHECK -----")
print(f"Raw min             : {int(chirps.min())}")
print(f"Raw max             : {int(chirps.max())}")
print(f"Raw mean            : {float(chirps.mean()):.2f}")

ADC_CENTER = 1 << (ADC_BITS - 1)
FS_PEAK = ADC_CENTER

chirps = chirps.astype(np.float32) - ADC_CENTER

# creates freq array with fft bin freq increment 0 2k 4k 6k .... fs/2
freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)

# range_m has range resolution steps.
# Radar resolution = c / 2B for LFM so 800 MHz makes 18.75cm resolution
# range_m steps are 18.75cm
if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

# max value of the range so max range
MAX_UNAMBIG_RANGE = np.max(range_m)

# if user selected value is smaller than max range max display is user range
# if user writes more than max range then normal max range is displayed.
MAX_RANGE_DISPLAY = min(MAX_RANGE_DISPLAY, MAX_UNAMBIG_RANGE) 

range_mask = range_m <= MAX_RANGE_DISPLAY
range_m_limited = range_m[range_mask]

print(f"MAX_UNAMBIG_RANGE   : {MAX_UNAMBIG_RANGE:.2f} m")
print(f"MAX_RANGE_DISPLAY   : {MAX_RANGE_DISPLAY:.2f} m")


w = np.hanning(SAMPLES_PER_CHIRP)
cg = np.sum(w) / SAMPLES_PER_CHIRP # coherent gain of the window the average of it.

# creates 2d pre-allocation buffer for waterfall plot
waterfall = np.full((DISPLAY_CPI_COUNT, len(range_m)), -160.0, dtype=np.float32)

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
    extent=[range_m_limited[0], range_m_limited[-1], 0, DISPLAY_CPI_COUNT]
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


for cpi_idx in range(0, FULL_CPI_COUNT, CHIRP_STEP):

    start = cpi_idx * CHIRPS_PER_CPI
    end = start + CHIRPS_PER_CPI

    # 2D array 128 chirps (rows) with each row has chirp adc samples
    chirps_cpi = chirps[start:end]

    # adc center removes bias but this takes any other dc bias could happen on signal
    if REMOVE_DC:
        chirps_cpi = chirps_cpi - np.mean(chirps_cpi, axis=1, keepdims=True)

    # apply window to the chirps
    chirps_cpi_w = chirps_cpi * w

    # chirps were centered by -2048, now scale it to -1,1 for 0dbfs reference
    chirps_cpi_fs = chirps_cpi_w / FS_PEAK

    # take fft over each chirp fast time processing
    chirps_fft = np.fft.rfft(chirps_cpi_fs, axis=1)

    # After fft data needs normalisation again which is done for windowed data
    chirps_fft = chirps_fft / (SAMPLES_PER_CHIRP * cg / 2)

    chirps_fft[:, :REMOVE_FIRST_N_BINS] = 0.0

    avg_power = np.mean(np.abs(chirps_fft) ** 2, axis=0)
    avg_range_dbfs = 10 * np.log10(avg_power + 1e-30)

    noise_mask = (
        (range_m > NOISE_RANGE_MIN) &
        (range_m < min(NOISE_RANGE_MAX, MAX_UNAMBIG_RANGE))
    )

    if np.any(noise_mask):
        noise_floor_dbfs = np.median(avg_range_dbfs[noise_mask])
    else:
        noise_floor_dbfs = np.median(avg_range_dbfs[10:])

    print(
        f"CPI {cpi_idx + 1:4d}/{FULL_CPI_COUNT}: "
        f"noise floor = {noise_floor_dbfs:.2f} dBFS/bin"
    )

    display_idx = cpi_idx // CHIRP_STEP
    waterfall[display_idx, :] = avg_range_dbfs

    # second fft over columns slow time fft for doppler
    doppler_fft = np.fft.fft(chirps_fft, axis=0)
    
    # reorder fft bins over dc zero frequency
    doppler_fft = np.fft.fftshift(doppler_fft, axes=0)

    rd_map_dbfs = 20 * np.log10(np.abs(doppler_fft) + 1e-30)
    rd_map_limited = rd_map_dbfs[:, range_mask]

    # noise floor of the 2D range-doppler map
    rd_noise_mask = (
        (range_m > NOISE_RANGE_MIN) &
        (range_m < min(NOISE_RANGE_MAX, MAX_UNAMBIG_RANGE))
    )

    if np.any(rd_noise_mask):
        rd_noise_floor_dbfs = np.median(rd_map_dbfs[:, rd_noise_mask])
    else:
        rd_noise_floor_dbfs = np.median(rd_map_dbfs[:, 10:])

    print(
        f"CPI {cpi_idx + 1:4d}/{FULL_CPI_COUNT}: "
        f"RD map noise floor = {rd_noise_floor_dbfs:.2f} dBFS/bin"
    )


    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    avg_range_dbfs_plot = np.convolve(avg_range_dbfs, kernel, mode="same")
    avg_range_limited = avg_range_dbfs_plot[range_mask]

    line.set_data(range_m_limited, avg_range_limited)

    ymax = np.max(avg_range_limited)
    noise = noise_floor_dbfs

    ax1.set_ylim(noise - 30, ymax + 5)

    ax1.set_title(
        f"Chirp Integrated Range Profile - CPI {cpi_idx + 1}/{FULL_CPI_COUNT} "
        f"| Noise {noise_floor_dbfs:.2f} dBFS/bin"
    )

    waterfall_limited = waterfall[:, range_mask]
    img.set_data(waterfall_limited)
    img.set_clim(np.max(waterfall_limited) - WATERFALL_NF_DB, np.max(waterfall_limited))

    img_rd.set_data(rd_map_limited)
    img_rd.set_clim(np.max(rd_map_limited) - DOPPLER_NF_DB, np.max(rd_map_limited))

    ax3.set_title(f"Range-Doppler Map - CPI {cpi_idx + 1}/{FULL_CPI_COUNT}")

    fig.canvas.draw_idle()
    plt.pause(0.01)

plt.ioff()
plt.show()