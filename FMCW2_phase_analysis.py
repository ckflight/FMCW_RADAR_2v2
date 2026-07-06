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

if OPERATING_SYSTEM == 1:
    BIN_FILE = "Radar_Records/data_record.bin"
    #BIN_FILE = "/home/ck/Desktop/flight_log.bin"
    #BIN_FILE = "fmcw2_bin_files/hwfir_terrace.bin"

elif OPERATING_SYSTEM == 2:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_TO_SHOW = 100

AUTO_PICK_BIN = True
FORCED_BIN = 41
IGNORE_FIRST_BINS = 0
AVG_CHIRPS_FOR_BIN_PICK = 128
DISPLAY_EVERY_N_CPI = 1

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


FS = FS_KHZ * 1000
num_chirps_expected = CPI_COUNTER * CHIRPS_PER_CPI

if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

if SAMPLES_PER_CHIRP <= 0:
    raise RuntimeError("SAMPLES_PER_CHIRP is zero")

if num_chirps_expected <= 0:
    raise RuntimeError("CPI_COUNTER or CHIRPS_PER_CPI is zero")


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
print(f"HEADER_SIZE         : {HEADER_SIZE if USE_SYNC_HEADERS else 0} words")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"ADC_BITS            : {ADC_BITS}")

print("\n----- TIMING -----")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")

print("\n----- CPI -----")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_RATE            : {CPI_RATE_HZ:.2f} Hz")
print(f"CPI_COUNTER         : {CPI_COUNTER}")
print(f"EXPECTED CHIRPS     : {num_chirps_expected}")


# -----------------------------
# Read ADC data with fixed-stride 4-word header
# -----------------------------
if USE_SYNC_HEADERS:
    words_per_chirp = SAMPLES_PER_CHIRP + HEADER_SIZE
else:
    words_per_chirp = SAMPLES_PER_CHIRP

bytes_to_read = num_chirps_expected * words_per_chirp * 2

raw_data = file_bytes[
    INFO_SECTOR_SIZE:
    INFO_SECTOR_SIZE + bytes_to_read
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

    chirps = chirps_raw[:, HEADER_SIZE:]

    print("\n----- SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"BAD HEADERS        : {len(bad_headers)}")
    print(f"UNUSED END WORDS   : {unused_words}")

    if len(bad_headers) > 0:
        print("First bad header indices:", bad_headers[:20])

else:
    chirps = chirps_raw

    print("\n----- NO SYNC -----")
    print(f"LOADED CHIRPS      : {len(chirps)}")
    print(f"UNUSED END WORDS   : {unused_words}")

num_chirps = len(chirps)

if num_chirps == 0:
    raise RuntimeError("No valid chirps found")


print("\n----- DATA -----")
print("Total words   :", len(data_u16))
print("Num chirps    :", num_chirps)
print("Samples/chirp :", SAMPLES_PER_CHIRP)


ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

chirps = chirps & ADC_MASK
chirps = chirps.astype(np.float32) - ADC_CENTER

# -----------------------------
# Keep only full CPIs
# -----------------------------
num_cpis = num_chirps // CHIRPS_PER_CPI
num_chirps_used = num_cpis * CHIRPS_PER_CPI

if num_cpis == 0:
    raise RuntimeError("Not enough chirps for one full CPI")

chirps = chirps[:num_chirps_used]
chirps_3d = chirps.reshape(num_cpis, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)

print("Num CPIs      :", num_cpis)
print("Chirps used   :", num_chirps_used)


# -----------------------------
# Fast-time window and FFT axes
# -----------------------------
w = np.hanning(SAMPLES_PER_CHIRP).astype(np.float32)

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)


# -----------------------------
# Complex range FFT cube
# -----------------------------
print("\nComputing complex FFTs...")

x = chirps_3d.copy()
x = x - np.mean(x, axis=2, keepdims=True)
x = x * w[None, None, :]

range_fft = np.fft.rfft(x, axis=2)


# -----------------------------
# Auto-pick strong range bin
# -----------------------------
flat_fft = range_fft.reshape(-1, range_fft.shape[2])
num_avg = min(AVG_CHIRPS_FOR_BIN_PICK, flat_fft.shape[0])

avg_mag = np.mean(np.abs(flat_fft[:num_avg]), axis=0)

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
# Extract tracked complex bin
# -----------------------------
tracked = range_fft[:, :, track_bin]

phase_chirp_wrapped = np.angle(tracked)
phase_chirp_unwrapped = np.unwrap(phase_chirp_wrapped, axis=1)

cpi_complex = np.mean(tracked, axis=1)
phase_cpi_wrapped = np.angle(cpi_complex)
phase_cpi_unwrapped = np.unwrap(phase_cpi_wrapped)

phase_step_chirp = np.angle(tracked[:, 1:] * np.conj(tracked[:, :-1]))
phase_step_cpi = np.angle(cpi_complex[1:] * np.conj(cpi_complex[:-1]))

chirp_idx = np.arange(CHIRPS_PER_CPI)
cpi_idx = np.arange(num_cpis)


# -----------------------------
# Display
# -----------------------------
plt.ion()
fig, ax = plt.subplots(2, 2, figsize=(14, 8))

line_cw, = ax[0, 0].plot([], [], lw=1.0)
line_cu, = ax[1, 0].plot([], [], lw=1.0)

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
# Live playback
# -----------------------------
for cpi_i in range(0, num_cpis, DISPLAY_EVERY_N_CPI):

    y_left_wrapped = phase_chirp_wrapped[cpi_i]
    y_left_unwrapped = phase_chirp_unwrapped[cpi_i]

    line_cw.set_data(chirp_idx, y_left_wrapped)
    line_cu.set_data(chirp_idx, y_left_unwrapped)

    x_right = cpi_idx[:cpi_i + 1]
    y_right_wrapped = phase_cpi_wrapped[:cpi_i + 1]
    y_right_unwrapped = phase_cpi_unwrapped[:cpi_i + 1]

    line_pw.set_data(x_right, y_right_wrapped)
    line_pu.set_data(x_right, y_right_unwrapped)

    fig.suptitle(
        f"Phase Stability Check | CPI {cpi_i + 1}/{num_cpis} | "
        f"bin={track_bin} | range={track_range_m:.2f} m",
        fontsize=14
    )

    fig.canvas.draw_idle()
    plt.pause(0.001)

plt.ioff()
plt.show()