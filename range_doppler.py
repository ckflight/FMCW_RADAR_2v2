import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_TO_SHOW = 200     # meters
RANGE_BIN_LIMIT = None      # set e.g. 512 if you want faster plotting
CPI_INDEX_TO_SHOW = 0       # which CPI to visualize

def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            (buf[offset + 3]))

def read_u16_be(buf, offset):
    return ((buf[offset] << 8) |
            (buf[offset + 1]))

# -----------------------------
# Read file once
# -----------------------------
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

if len(file_bytes) < INFO_SECTOR_SIZE:
    raise ValueError("File is smaller than 512-byte info sector")

info = file_bytes[:INFO_SECTOR_SIZE]
raw_data = file_bytes[INFO_SECTOR_SIZE:]

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

FS = FS_KHZ * 1000
SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW = SWEEP_BW_SCALED * 1e6

PRF = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

print("----- SYSTEM -----")
print(f"FS                  : {FS/1e6:.2f} MHz")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"PRF                 : {PRF:.2f} Hz")

# -----------------------------
# Read ADC data
# -----------------------------
data = np.frombuffer(raw_data, dtype='<u2')

num_chirps = len(data) // SAMPLES_PER_CHIRP
data = data[:num_chirps * SAMPLES_PER_CHIRP]
chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

print("\n----- DATA -----")
print("Total samples :", len(data))
print("Num chirps    :", num_chirps)
print("Samples/chirp :", SAMPLES_PER_CHIRP)

# unsigned ADC -> centered float
chirps = chirps.astype(np.float32) - 32768.0

# -----------------------------
# Split into full CPIs
# -----------------------------
num_cpis = num_chirps // CHIRPS_PER_CPI
if num_cpis == 0:
    raise ValueError("Not enough chirps for one CPI")

chirps = chirps[:num_cpis * CHIRPS_PER_CPI]
cpis = chirps.reshape(num_cpis, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)

print("Num CPIs      :", num_cpis)

if CPI_INDEX_TO_SHOW >= num_cpis:
    raise ValueError("CPI_INDEX_TO_SHOW exceeds available CPI count")

cpi = cpis[CPI_INDEX_TO_SHOW]   # shape: [chirps_per_cpi, samples_per_chirp]

# -----------------------------
# Windowing
# -----------------------------
w_range = np.hanning(SAMPLES_PER_CHIRP).astype(np.float32)
w_dopp = np.hanning(CHIRPS_PER_CPI).astype(np.float32)

# remove per-chirp DC
cpi = cpi - np.mean(cpi, axis=1, keepdims=True)

# -----------------------------
# Range FFT
# -----------------------------
cpi_range_win = cpi * w_range[None, :]
range_fft = np.fft.rfft(cpi_range_win, axis=1)

# suppress leakage/DC bins
range_fft[:, :5] = 0

# optional crop range bins for speed
if RANGE_BIN_LIMIT is not None:
    range_fft = range_fft[:, :RANGE_BIN_LIMIT]

num_range_bins = range_fft.shape[1]

# -----------------------------
# Doppler FFT
# -----------------------------
range_fft_dopp_win = range_fft * w_dopp[:, None]
rdm = np.fft.fft(range_fft_dopp_win, axis=0)
rdm = np.fft.fftshift(rdm, axes=0)

rdm_mag = 20 * np.log10(np.abs(rdm) + 1e-12)

# normalize for display
rdm_mag = rdm_mag - np.max(rdm_mag)

# -----------------------------
# Axes
# -----------------------------
freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)[:num_range_bins]
if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(num_range_bins)

doppler_hz = np.fft.fftshift(np.fft.fftfreq(CHIRPS_PER_CPI, d=1.0 / PRF))

# if you want velocity axis:
fc = SWEEP_START + SWEEP_BW / 2.0
lam = 3e8 / fc
velocity_m_s = doppler_hz * lam / 2.0

# -----------------------------
# Plot range profile average over CPI
# -----------------------------
avg_range_profile = 20 * np.log10(np.mean(np.abs(range_fft), axis=0) + 1e-12)
avg_range_profile = avg_range_profile - np.max(avg_range_profile)

plt.figure(figsize=(10, 5))
plt.plot(range_m, avg_range_profile)
plt.xlabel("Range (m)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Average Range Profile - CPI {CPI_INDEX_TO_SHOW}")
plt.grid(True)
plt.xlim(0, min(MAX_RANGE_TO_SHOW, np.max(range_m)))
plt.ylim(-80, 5)
plt.show()

# -----------------------------
# Plot Range-Doppler map
# -----------------------------
plt.figure(figsize=(10, 6))
plt.imshow(
    rdm_mag,
    aspect='auto',
    origin='lower',
    extent=[range_m[0], range_m[-1], velocity_m_s[0], velocity_m_s[-1]]
)
plt.xlabel("Range (m)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Range-Doppler Map - CPI {CPI_INDEX_TO_SHOW}")
plt.colorbar(label="dB")
plt.clim(-60, 0)
plt.xlim(0, min(MAX_RANGE_TO_SHOW, np.max(range_m)))
plt.show()