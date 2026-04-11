import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE        = 512
MAX_RANGE_DISPLAY       = 20 # range upper plot limit in meters 

def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            (buf[offset + 3]))

def read_u16_be(buf, offset):
    return ((buf[offset] << 8) |
            (buf[offset + 1]))


# -----------------------------
# Read the whole .bin file once first 512 byte is info, rest is data
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

CHIRPS_PER_CPI          = read_u16_be(info, idx); idx += 2
CPI_COUNTER             = read_u32_be(info, idx); idx += 4

# This is the amount of byte logged
NUM_OF_BYTES_LOGGED = int(CPI_COUNTER * CHIRPS_PER_CPI * SAMPLES_PER_CHIRP * (ADC_BITS / 8)) 

# Instead of buffering 500 Megabyte whole .bin file, take amount of byte written.
raw_data = file_bytes[INFO_SECTOR_SIZE: INFO_SECTOR_SIZE + NUM_OF_BYTES_LOGGED]

# -----------------------------
# Reconstruct user-friendly values
# -----------------------------
FS = FS_KHZ * 1000
SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6

# -----------------------------
# Derived values
# -----------------------------
CONFIGURED_PRF_HZ = 0.0
if (SWEEP_TIME_US + SWEEP_GAP_US) > 0:
    CONFIGURED_PRF_HZ = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

MEASURED_CHIRP_RATE_HZ = 0.0
if CHIRP_END_TIMER_US > 0:
    MEASURED_CHIRP_RATE_HZ = 1e6 / CHIRP_END_TIMER_US

CPI_RATE_HZ = 0.0
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    CPI_RATE_HZ = 1e6 / (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US)

BYTES_PER_SAMPLE = 2 if USB_DATA_TYPE == 1 else 1
BYTES_PER_CHIRP = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE 
BYTES_PER_CPI = CHIRPS_PER_CPI * BYTES_PER_CHIRP

CONFIGURED_DATA_RATE_MBPS = (BYTES_PER_CHIRP * CONFIGURED_PRF_HZ) / 1e6

CARD_WRITE_SPEED_MBPS = 0.0
if CARD_WRITE_END_TIMER_US > 0:
    CARD_WRITE_SPEED_MBPS = BYTES_PER_CPI / CARD_WRITE_END_TIMER_US

APPROX_RECORD_TIME_FROM_COUNTER = 0.0
if CONFIGURED_PRF_HZ > 0:
    APPROX_RECORD_TIME_FROM_COUNTER = RECORD_COUNTER / CONFIGURED_PRF_HZ

TOTAL_CPI_TIME_S = (CPI_COUNTER * CPI_END_TIMER_US) / 1e6
TOTAL_CARD_WRITE_TIME_S = (CPI_COUNTER * CARD_WRITE_END_TIMER_US) / 1e6
print("\n----- SYSTEM -----")
print(f"FS                  : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"HZ_PER_M            : {HZ_PER_M}")

print("\n----- TIMING -----")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")

print("\n----- CPI -----")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_RATE            : {CPI_RATE_HZ:.2f} Hz")
print(f"CPI_COUNTER         : {CPI_COUNTER}")

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")

# PROCESSING PART
data_16bit = np.frombuffer(raw_data, dtype='<u2') # interpret raw_data as each 2 byte is a 16 bit sample
data_16bit = data_16bit.astype(np.float32)
data_16bit -= 32768.0 # remove bias float around 0v

num_chirps = CPI_COUNTER * CHIRPS_PER_CPI

# 16 bit record array is converted to number of chirps of whole record row with each row has adc samples as columns
chirps = data_16bit.reshape(num_chirps, SAMPLES_PER_CHIRP) 


# This part takes each cpi as a 2D array so 128 chirps x 930 sample on each row.
# Then takes fft of this 2D array.
# And creates sum over 2D array to create 1D array. 128 chirp is averaged and power is calculated. 
# As a result peaks becomes higher noise reduced.

# for range
freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
range_m = freq_hz / HZ_PER_M

# for waterfall
waterfall = np.zeros((CPI_COUNTER, len(range_m)), dtype=np.float32)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.subplots_adjust(hspace=0.28)

# Top → 1D range profile
line, = ax1.plot([], [])
ax1.set_xlabel("Range (m)")
ax1.set_ylabel("Power")
ax1.set_title("Range Profile (CPI)")
ax1.grid(True)

# Bottom → waterfall
waterfall = np.zeros((CPI_COUNTER, len(range_m)), dtype=np.float32)

img = ax2.imshow(
    np.zeros_like(waterfall),
    aspect='auto',
    origin='lower',
    extent=[range_m[0], range_m[-1], 0, CPI_COUNTER]
)
ax2.set_xlabel("Range (m)")
ax2.set_ylabel("CPI index")
ax2.set_title("Waterfall")

for cpi_idx in range(CPI_COUNTER):

    # Extract one CPI
    start = cpi_idx * CHIRPS_PER_CPI
    end   = (cpi_idx + 1) * CHIRPS_PER_CPI

    # 2d array of chirps of each cpi so 128 x 930 array
    chirps_cpi = chirps[start:end]

    # FFT over each chirp (fast-time) 
    # 128 x 930 --FFT--> 128 x 446
    chirps_fft = np.fft.rfft(chirps_cpi, axis=1)

    # Non-coherent integration (power averaging)
    avg_range = np.mean(np.abs(chirps_fft)**2, axis=0)

    # add to waterfall
    waterfall[cpi_idx, :] = avg_range

    # --- update 1D plot ---
    line.set_data(range_m, avg_range)
    ax1.set_xlim(0, MAX_RANGE_DISPLAY)
    ax1.set_ylim(np.min(avg_range), np.max(avg_range) * 1.1)
    ax1.set_title(f"Range Profile - CPI {cpi_idx+1}/{CPI_COUNTER}")

    # --- update waterfall ---
    waterfall_db = 10 * np.log10(waterfall + 1e-12)
    img.set_data(waterfall_db)

    # update color scaling so contrast becomes visible
    img.set_clim(np.max(waterfall_db) - 20, np.max(waterfall_db))
    ax2.set_xlim(0, MAX_RANGE_DISPLAY)
    ax2.set_ylim(0, CPI_COUNTER)

    fig.canvas.draw_idle()
    plt.pause(0.04)

plt.ioff()
plt.show()
