import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 1   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
else:
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"

INFO_SECTOR_SIZE  = 512
DISPLAY_STEP      = 20
MAX_RANGE_TO_SHOW = 50   # meters, change this value


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
# Validate
# -----------------------------
if ADC_BITS not in (10, 12, 14, 16):
    raise ValueError(f"Unsupported ADC_BITS = {ADC_BITS}")

FS = FS_KHZ * 1000
num_chirps = CPI_COUNTER * CHIRPS_PER_CPI

if num_chirps <= 0:
    raise ValueError("num_chirps is zero")

if SAMPLES_PER_CHIRP <= 0:
    raise ValueError("SAMPLES_PER_CHIRP is zero")


# -----------------------------
# Derived values
# -----------------------------
SWEEP_TIME  = SWEEP_TIME_US * 1e-6
SWEEP_GAP   = SWEEP_GAP_US * 1e-6
SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW    = SWEEP_BW_SCALED * 1e6

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
# ADC byte size
# Important:
# STM32 ADC DMA stores 10/12/14/16-bit samples as uint16_t.
# So file sample size is always 2 bytes/sample.
# -----------------------------
BYTES_PER_SAMPLE = 2
BYTES_PER_CHIRP = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE
BYTES_PER_CPI = CHIRPS_PER_CPI * BYTES_PER_CHIRP
NUM_OF_BYTES_LOGGED = num_chirps * BYTES_PER_CHIRP

CONFIGURED_DATA_RATE_MBPS = (BYTES_PER_CHIRP * CONFIGURED_PRF_HZ) / 1e6

CARD_WRITE_SPEED_MBPS = 0.0
if CARD_WRITE_END_TIMER_US > 0:
    CARD_WRITE_SPEED_MBPS = BYTES_PER_CPI / CARD_WRITE_END_TIMER_US


# -----------------------------
# Print info
# -----------------------------
print("\n----- SYSTEM -----")
print(f"FS                  : {FS / 1e6:.3f} MHz")
print(f"ADC_BITS            : {ADC_BITS}")
print(f"SAMPLES_PER_CHIRP   : {SAMPLES_PER_CHIRP}")
print(f"HZ_PER_M            : {HZ_PER_M}")
print(f"MAX_RANGE_TO_SHOW   : {MAX_RANGE_TO_SHOW} m")

print("\n----- TIMING -----")
print(f"SWEEP_TIME          : {SWEEP_TIME_US} us")
print(f"SWEEP_GAP           : {SWEEP_GAP_US} us")
print(f"CONFIGURED_PRF      : {CONFIGURED_PRF_HZ:.2f} Hz")
print(f"MEASURED_CHIRP_RATE : {MEASURED_CHIRP_RATE_HZ:.2f} Hz")

print("\n----- CPI -----")
print(f"CHIRPS_PER_CPI      : {CHIRPS_PER_CPI}")
print(f"CPI_COUNTER         : {CPI_COUNTER}")
print(f"NUM_CHIRPS          : {num_chirps}")
print(f"CPI_RATE            : {CPI_RATE_HZ:.2f} Hz")

print("\n----- DATA -----")
print(f"BYTES_PER_SAMPLE    : {BYTES_PER_SAMPLE}")
print(f"BYTES_PER_CHIRP     : {BYTES_PER_CHIRP}")
print(f"DATA_RATE           : {CONFIGURED_DATA_RATE_MBPS:.2f} MB/s")
print(f"WRITE_SPEED         : {CARD_WRITE_SPEED_MBPS:.2f} MB/s")


# -----------------------------
# Read ADC data
# -----------------------------
raw_data = file_bytes[INFO_SECTOR_SIZE : INFO_SECTOR_SIZE + NUM_OF_BYTES_LOGGED]

if len(raw_data) < NUM_OF_BYTES_LOGGED:
    raise ValueError(
        f"Not enough data bytes: got {len(raw_data)}, expected {NUM_OF_BYTES_LOGGED}"
    )

adc_u16 = np.frombuffer(raw_data, dtype="<u2")

SYNC = 0xC8C8

chirps = []

sync_idx = np.where(adc_u16[:-1] == SYNC)[0]

for i in sync_idx:

    if adc_u16[i + 1] == SYNC:

        chirp = adc_u16[i + 2 : i + 2 + SAMPLES_PER_CHIRP]

        if len(chirp) == SAMPLES_PER_CHIRP:
            chirps.append(chirp)

        i += SAMPLES_PER_CHIRP + 2

    else:
        i += 1

chirps = np.array(chirps)

num_chirps = len(chirps)

print(f"SYNCED CHIRPS : {num_chirps}")


# -----------------------------
# Generic ADC bit handling
# -----------------------------
ADC_MASK = (1 << ADC_BITS) - 1
ADC_CENTER = float(1 << (ADC_BITS - 1))

adc_raw = chirps & ADC_MASK
adc_centered = chirps.astype(np.float32) - ADC_CENTER

chirps = adc_centered

print("\n----- ADC CHECK -----")
print(f"Raw min       : {adc_raw.min()}")
print(f"Raw max       : {adc_raw.max()}")
print(f"Centered min  : {adc_centered.min():.1f}")
print(f"Centered max  : {adc_centered.max():.1f}")
print(f"Centered mean : {adc_centered.mean():.2f}")


# -----------------------------
# FFT precompute
# -----------------------------
w = np.hanning(SAMPLES_PER_CHIRP)
cg = np.sum(w) / SAMPLES_PER_CHIRP
FS_PEAK = ADC_CENTER

freq_hz = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS)
freq_khz = freq_hz / 1e3

if HZ_PER_M > 0:
    range_m = freq_hz / HZ_PER_M
else:
    range_m = np.arange(len(freq_hz), dtype=np.float32)

range_mask = range_m <= MAX_RANGE_TO_SHOW
range_m_plot = range_m[range_mask]

spectra = np.empty((num_chirps, len(freq_hz)), dtype=np.float32)

for i in range(num_chirps):
    x = chirps[i]
    x = x - np.mean(x)

    xw = x * w
    xw_fs = xw / FS_PEAK

    X = np.fft.rfft(xw_fs)

    mag_dbfs = 20.0 * np.log10(
        (np.abs(X) / (SAMPLES_PER_CHIRP * cg / 2.0)) + 1e-20
    )

    mag_dbfs[:5] = np.min(mag_dbfs)
    spectra[i] = mag_dbfs


# -----------------------------
# Averaged FFT
# -----------------------------
N_AVG = min(CHIRPS_PER_CPI, num_chirps)
avg_mag_dbfs = np.mean(spectra[:N_AVG], axis=0)

# -----------------------------
# Average noise floor
# -----------------------------
avg_limited = avg_mag_dbfs[range_mask]

# remove strongest peaks for cleaner floor estimate
sorted_mag = np.sort(avg_limited)

# ignore top 5% bins
noise_bins = sorted_mag[:int(len(sorted_mag) * 0.95)]

AVG_NOISE_FLOOR_DBFS = np.mean(noise_bins)

print(f"\nAVERAGE NOISE FLOOR : {AVG_NOISE_FLOOR_DBFS:.2f} dBFS/bin")

print("\nTop peaks from averaged FFT:")
avg_limited = avg_mag_dbfs[range_mask]
range_limited = range_m[range_mask]
freq_limited = freq_khz[range_mask]

peak_indices = np.argsort(avg_limited)[-10:][::-1]

for k in peak_indices:
    print(
        f"freq={freq_limited[k]:10.2f} kHz  "
        f"range={range_limited[k]:8.2f} m  "
        f"mag={avg_limited[k]:7.2f} dBFS"
    )


# -----------------------------
# Animation
# -----------------------------
plt.ion()

fig_anim, ax_anim = plt.subplots(figsize=(10, 5))

spectra_plot = spectra[:, range_mask]

line, = ax_anim.plot(range_m_plot, spectra_plot[0])

ax_anim.set_xlabel("Range (m)")
ax_anim.set_ylabel("Magnitude (dBFS)")
ax_anim.set_title("Range Profile - Chirp 0")
ax_anim.grid(True)

ax_anim.set_xlim(0, MAX_RANGE_TO_SHOW)

global_ymax = np.max(spectra_plot)
ax_anim.set_ylim(global_ymax - 80, global_ymax + 5)

fig_anim.tight_layout()
fig_anim.show()

for i in range(0, num_chirps, DISPLAY_STEP):
    line.set_ydata(spectra_plot[i])
    ax_anim.set_title(f"Range Profile - Chirp {i}")
    fig_anim.canvas.draw_idle()
    plt.pause(0.001)

plt.ioff()
plt.show()