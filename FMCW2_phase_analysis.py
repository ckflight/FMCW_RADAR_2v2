import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_TO_SHOW = 100

# -----------------------------
# phase analysis config
# -----------------------------
AUTO_PICK_BIN = True
FORCED_BIN = 54          # used if AUTO_PICK_BIN = False
IGNORE_FIRST_BINS = 5
AVG_CHIRPS_FOR_BIN_PICK = 256
DISPLAY_EVERY_N_CPI = 1  # 1 = show every CPI, 2 = every 2nd CPI ...

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

# keep your override
SAMPLES_PER_CHIRP = 900

# -----------------------------
# Reconstruct user-friendly values
# -----------------------------
FS = FS_KHZ * 1000
SWEEP_TIME = SWEEP_TIME_US * 1e-6
SWEEP_GAP = SWEEP_GAP_US * 1e-6

CONFIGURED_PRF_HZ = 0.0
if (SWEEP_TIME_US + SWEEP_GAP_US) > 0:
    CONFIGURED_PRF_HZ = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

MEASURED_CHIRP_RATE_HZ = 0.0
if CHIRP_END_TIMER_US > 0:
    MEASURED_CHIRP_RATE_HZ = 1e6 / CHIRP_END_TIMER_US

CPI_RATE_HZ = 0.0
if (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US) > 0:
    CPI_RATE_HZ = 1e6 / (CPI_END_TIMER_US + CARD_WRITE_END_TIMER_US)

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

# -----------------------------
# Read ADC data
# -----------------------------
data = np.frombuffer(raw_data, dtype='<u2')

num_chirps = CPI_COUNTER * CHIRPS_PER_CPI
data = data[:num_chirps * SAMPLES_PER_CHIRP]
chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

print("\n----- DATA -----")
print("Total samples :", len(data))
print("Num chirps    :", num_chirps)
print("Samples/chirp :", SAMPLES_PER_CHIRP)

# unsigned ADC -> centered float
chirps = chirps.astype(np.float32) - 32768.0

# -----------------------------
# keep only full CPIs
# -----------------------------
num_cpis = num_chirps // CHIRPS_PER_CPI
num_chirps_used = num_cpis * CHIRPS_PER_CPI
chirps = chirps[:num_chirps_used]
chirps_3d = chirps.reshape(num_cpis, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)

print("Num CPIs      :", num_cpis)
print("Chirps used   :", num_chirps_used)

# -----------------------------
# fast-time window and FFT axes
# -----------------------------
w = np.hanning(SAMPLES_PER_CHIRP).astype(np.float32)

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

# -----------------------------
# complex range FFT cube
# -----------------------------
print("\nComputing complex FFTs...")

x = chirps_3d.copy()
x = x - np.mean(x, axis=2, keepdims=True)          # DC remove each chirp
x = x * w[None, None, :]
range_fft = np.fft.rfft(x, axis=2)                 # keep complex values

# -----------------------------
# auto-pick strong range bin
# -----------------------------
avg_mag = np.mean(np.abs(range_fft.reshape(-1, range_fft.shape[2]))[:AVG_CHIRPS_FOR_BIN_PICK], axis=0)

valid = np.where(range_m <= MAX_RANGE_TO_SHOW)[0]
valid = valid[valid >= IGNORE_FIRST_BINS]

if len(valid) == 0:
    raise ValueError("No valid range bins")

if AUTO_PICK_BIN:
    track_bin = valid[np.argmax(avg_mag[valid])]
else:
    track_bin = FORCED_BIN

track_range_m = range_m[track_bin]
track_freq_hz = freq_hz[track_bin]

print("\n----- TRACKED BIN -----")
print(f"track_bin    : {track_bin}")
print(f"track_freq   : {track_freq_hz/1e3:.2f} kHz")
print(f"track_range  : {track_range_m:.2f} m")

# -----------------------------
# extract tracked complex bin
# -----------------------------
# shape = [num_cpis, CHIRPS_PER_CPI]
tracked = range_fft[:, :, track_bin]

# chirp-to-chirp phase inside each CPI
phase_chirp_wrapped = np.angle(tracked)
phase_chirp_unwrapped = np.unwrap(phase_chirp_wrapped, axis=1)

# one complex value per CPI
cpi_complex = np.mean(tracked, axis=1)
phase_cpi_wrapped = np.angle(cpi_complex)
phase_cpi_unwrapped = np.unwrap(phase_cpi_wrapped)

# optional: phase step diagnostics
phase_step_chirp = np.angle(tracked[:, 1:] * np.conj(tracked[:, :-1]))
phase_step_cpi = np.angle(cpi_complex[1:] * np.conj(cpi_complex[:-1]))

# -----------------------------
# x axes
# -----------------------------
chirp_idx = np.arange(CHIRPS_PER_CPI)
cpi_idx = np.arange(num_cpis)

# -----------------------------
# display
# -----------------------------
plt.ion()
fig, ax = plt.subplots(2, 2, figsize=(14, 8))

# left side = current CPI chirp-to-chirp
line_cw, = ax[0, 0].plot([], [], lw=1.0)
line_cu, = ax[1, 0].plot([], [], lw=1.0)

# right side = CPI-to-CPI up to current point
line_pw, = ax[0, 1].plot([], [], lw=1.0)
line_pu, = ax[1, 1].plot([], [], lw=1.0)

ax[0, 0].set_title("Chirp-to-Chirp Phase (wrapped)")
ax[1, 0].set_title("Chirp-to-Chirp Phase (unwrapped)")
ax[0, 1].set_title("CPI-to-CPI Phase (wrapped)")
ax[1, 1].set_title("CPI-to-CPI Phase (unwrapped)")

ax[0, 0].set_xlabel("Chirp index in CPI")
ax[1, 0].set_xlabel("Chirp index in CPI")
ax[0, 1].set_xlabel("CPI index")
ax[1, 1].set_xlabel("CPI index")

ax[0, 0].set_ylabel("Phase (rad)")
ax[1, 0].set_ylabel("Phase (rad)")
ax[0, 1].set_ylabel("Phase (rad)")
ax[1, 1].set_ylabel("Phase (rad)")

for a in ax.flatten():
    a.grid(True)

# fixed limits
ax[0, 0].set_xlim(0, CHIRPS_PER_CPI - 1)
ax[1, 0].set_xlim(0, CHIRPS_PER_CPI - 1)
ax[0, 1].set_xlim(0, num_cpis - 1)
ax[1, 1].set_xlim(0, num_cpis - 1)

ax[0, 0].set_ylim(-np.pi - 0.2, np.pi + 0.2)
ax[0, 1].set_ylim(-np.pi - 0.2, np.pi + 0.2)

chirp_unwrap_min = np.min(phase_chirp_unwrapped)
chirp_unwrap_max = np.max(phase_chirp_unwrapped)
if np.isclose(chirp_unwrap_min, chirp_unwrap_max):
    chirp_unwrap_min -= 1
    chirp_unwrap_max += 1
ax[1, 0].set_ylim(chirp_unwrap_min - 0.2, chirp_unwrap_max + 0.2)

cpi_unwrap_min = np.min(phase_cpi_unwrapped)
cpi_unwrap_max = np.max(phase_cpi_unwrapped)
if np.isclose(cpi_unwrap_min, cpi_unwrap_max):
    cpi_unwrap_min -= 1
    cpi_unwrap_max += 1
ax[1, 1].set_ylim(cpi_unwrap_min - 0.2, cpi_unwrap_max + 0.2)

fig.suptitle("Phase Stability Check", fontsize=14)
fig.tight_layout()

# -----------------------------
# live playback
# -----------------------------
for cpi_i in range(0, num_cpis, DISPLAY_EVERY_N_CPI):
    # left: current CPI chirp-to-chirp
    y_left_wrapped = phase_chirp_wrapped[cpi_i]
    y_left_unwrapped = phase_chirp_unwrapped[cpi_i]

    line_cw.set_data(chirp_idx, y_left_wrapped)
    line_cu.set_data(chirp_idx, y_left_unwrapped)

    # right: CPI-to-CPI up to current CPI
    x_right = cpi_idx[:cpi_i + 1]
    y_right_wrapped = phase_cpi_wrapped[:cpi_i + 1]
    y_right_unwrapped = phase_cpi_unwrapped[:cpi_i + 1]

    line_pw.set_data(x_right, y_right_wrapped)
    line_pu.set_data(x_right, y_right_unwrapped)

    fig.suptitle(
        f"Phase Stability Check | CPI {cpi_i+1}/{num_cpis} | "
        f"bin={track_bin} | range={track_range_m:.2f} m",
        fontsize=14
    )

    fig.canvas.draw_idle()
    plt.pause(0.001)

plt.ioff()
plt.show()