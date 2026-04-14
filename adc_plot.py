import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 2   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\terrace1.bin"

INFO_SECTOR_SIZE        = 512
MAX_RANGE_DISPLAY       = 70 # range upper plot limit in meters 

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
print(f"TX_POWER            : {TX_POWER_DBM}")
print(f"TX_VOLT             : {TX_POWER_DBM_VOLT}")
print(f"SWEEP_START         : {SWEEP_START}")
print(f"SWEEP_BW            : {SWEEP_BW}")

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

# -----------------------------
# Extract RAW ADC only
# -----------------------------
raw_data = file_bytes[INFO_SECTOR_SIZE:]

data = np.frombuffer(raw_data, dtype='<u2').astype(np.float32)
data -= 32768.0

num_chirps = CPI_COUNTER * CHIRPS_PER_CPI
data = data[:num_chirps * SAMPLES_PER_CHIRP]

chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

# -----------------------------
# Plot
# -----------------------------
plt.ion
fig, ax = plt.subplots(figsize=(10,8))


for cpi_idx in range(CPI_COUNTER):
    
    start = CHIRPS_PER_CPI * cpi_idx
    end   = CHIRPS_PER_CPI * (cpi_idx + 1)
    
    chirps_cpi = chirps[start : end]
    
    average_chirps = np.mean(chirps_cpi, axis = 0)

    ax.clear()
    ax.plot(average_chirps)#, label=f"chirps {idx}", alpha=0.6)

    ax.set_title(f"Raw ADC CPI {cpi_idx+1}/{CPI_COUNTER}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("ADC Value")
    ax.grid(True)
    
    plt.pause(0.05)

plt.ioff
plt.show()