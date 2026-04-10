import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512
DISPLAY_STEP = 5            # animate every Nth chirp
MAX_RANGE_TO_SHOW = 100     # meters
AVG_CHIRPS = 64             # number of chirps for averaged FFT

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
if CPI_END_TIMER_US > 0:
    CPI_RATE_HZ = 1e6 / CPI_END_TIMER_US

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

print("\n----- DATA -----")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")

print("\n----- SD WRITE -----")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")

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

# unsigned ADC -> centered signed-like float
chirps = chirps.astype(np.float32) - 32768.0

# -----------------------------
# FFT precompute once
# -----------------------------
w = np.hanning(SAMPLES_PER_CHIRP)
cg = np.sum(w) / SAMPLES_PER_CHIRP
FS_PEAK = 2**(ADC_BITS - 1)

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
freq_khz = freq_hz / 1e3

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

spectra = np.empty((num_chirps, len(freq_hz)), dtype=np.float32)

for i in range(num_chirps):
    x = chirps[i].copy()
    x = x - np.mean(x)
    xw = x * w
    xw_fs = xw / FS_PEAK
    X = np.fft.rfft(xw_fs)
    mag_dbfs = 20 * np.log10((np.abs(X) / (SAMPLES_PER_CHIRP * cg / 2)) + 1e-20)
    mag_dbfs[:5] = np.min(mag_dbfs)   # suppress DC / leakage bins
    spectra[i] = mag_dbfs

# -----------------------------
# Averaged FFT
# -----------------------------
N_AVG = min(AVG_CHIRPS, num_chirps)
avg_mag_dbfs = np.mean(spectra[:N_AVG], axis=0)

print("\nTop peaks from averaged FFT:")
peak_indices = np.argsort(avg_mag_dbfs)[-10:][::-1]
for k in peak_indices:
    print(f"bin={k:4d}  freq={freq_khz[k]:10.2f} kHz  range={range_m[k]:8.2f} m  mag={avg_mag_dbfs[k]:7.2f} dBFS")

# -----------------------------
# Animation first, finite only
# -----------------------------
plt.ion()
fig_anim, ax_anim = plt.subplots(figsize=(10, 5))
line, = ax_anim.plot(range_m, spectra[0])
ax_anim.set_xlabel("Range (m)")
ax_anim.set_ylabel("Magnitude (dBFS)")
ax_anim.set_title("Range Profile - Chirp 0")
ax_anim.grid(True)
ax_anim.set_xlim(0, min(MAX_RANGE_TO_SHOW, np.max(range_m)))

global_ymax = np.max(spectra)
ax_anim.set_ylim(global_ymax - 80, global_ymax + 5)
fig_anim.tight_layout()
fig_anim.show()

for i in range(0, num_chirps, DISPLAY_STEP):
    line.set_ydata(spectra[i])
    ax_anim.set_title(f"Range Profile - Chirp {i}")
    fig_anim.canvas.draw_idle()
    plt.pause(0.001)

# leave the last displayed chirp on screen and stop updating
plt.ioff()

# -----------------------------
# Static plots
# -----------------------------
fig_avg, ax_avg = plt.subplots(figsize=(10, 5))
ax_avg.plot(range_m, avg_mag_dbfs)
ax_avg.set_xlabel("Range (m)")
ax_avg.set_ylabel("Magnitude (dBFS)")
ax_avg.set_title(f"Averaged FFT ({N_AVG} chirps)")
ax_avg.grid(True)
ymax = np.max(avg_mag_dbfs)
ax_avg.set_ylim(ymax - 80, ymax + 5)
ax_avg.set_xlim(0, min(MAX_RANGE_TO_SHOW, np.max(range_m)))
fig_avg.tight_layout()

fig_rt, ax_rt = plt.subplots(figsize=(10, 6))
im = ax_rt.imshow(
    spectra,
    aspect='auto',
    origin='lower',
    extent=[range_m[0], range_m[-1], 0, num_chirps]
)
ax_rt.set_xlabel("Range (m)")
ax_rt.set_ylabel("Chirp index")
ax_rt.set_title("Range-Time Intensity")
ax_rt.set_xlim(0, min(MAX_RANGE_TO_SHOW, np.max(range_m)))
fig_rt.colorbar(im, ax=ax_rt, label="dBFS")
fig_rt.tight_layout()

# one final blocking show
plt.show()