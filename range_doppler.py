import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
BIN_FILE = "/home/ck/Desktop/flight_log.bin"

INFO_SECTOR_SIZE = 512
MAX_RANGE_TO_SHOW = 100
RANGE_BIN_LIMIT = 512      # speed vs resolution
DISPLAY_STEP_CPI = 1       # skip CPIs if needed (2,4,...)
PAUSE_TIME = 0.1          # playback speed

# -----------------------------
# Helpers
# -----------------------------
def read_u32_be(buf, offset):
    return ((buf[offset] << 24) |
            (buf[offset + 1] << 16) |
            (buf[offset + 2] << 8) |
            (buf[offset + 3]))

def read_u16_be(buf, offset):
    return ((buf[offset] << 8) |
            (buf[offset + 1]))

# -----------------------------
# Read file
# -----------------------------
with open(BIN_FILE, "rb") as f:
    file_bytes = f.read()

info = file_bytes[:INFO_SECTOR_SIZE]
raw_data = file_bytes[INFO_SECTOR_SIZE:]

# -----------------------------
# Decode info
# -----------------------------
idx = 0
idx += 4  # RECORD_COUNTER
idx += 4  # RECORD_TIME
SWEEP_TIME_US = read_u32_be(info, idx); idx += 4
SWEEP_GAP_US  = read_u32_be(info, idx); idx += 4
SWEEP_START_SCALED = read_u32_be(info, idx); idx += 4
SWEEP_BW_SCALED    = read_u32_be(info, idx); idx += 4
FS_KHZ             = read_u32_be(info, idx); idx += 4
SAMPLES_PER_CHIRP  = read_u32_be(info, idx); idx += 4

idx += 3  # skip TX params

HZ_PER_M = read_u32_be(info, idx); idx += 4

idx += 4  # DATA_LOG etc

idx += 12  # timers

CHIRPS_PER_CPI = read_u16_be(info, idx); idx += 2
idx += 4  # CPI_COUNTER

# -----------------------------
# Derived values
# -----------------------------
FS = FS_KHZ * 1000
PRF = 1e6 / (SWEEP_TIME_US + SWEEP_GAP_US)

SWEEP_START = SWEEP_START_SCALED * 1e7
SWEEP_BW    = SWEEP_BW_SCALED * 1e6

fc = SWEEP_START + SWEEP_BW / 2
lam = 3e8 / fc

print("FS:", FS)
print("Samples/chirp:", SAMPLES_PER_CHIRP)
print("Chirps/CPI:", CHIRPS_PER_CPI)
print("PRF:", PRF)

# -----------------------------
# Load ADC data
# -----------------------------
data = np.frombuffer(raw_data, dtype='<u2')

num_chirps = len(data) // SAMPLES_PER_CHIRP
data = data[:num_chirps * SAMPLES_PER_CHIRP]

chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

# center ADC
chirps = chirps.astype(np.float32) - 32768.0

# -----------------------------
# Split CPIs
# -----------------------------
num_cpis = num_chirps // CHIRPS_PER_CPI
chirps = chirps[:num_cpis * CHIRPS_PER_CPI]

cpis = chirps.reshape(num_cpis, CHIRPS_PER_CPI, SAMPLES_PER_CHIRP)

print("Num CPIs:", num_cpis)

# -----------------------------
# FFT preparation
# -----------------------------
w_range = np.hanning(SAMPLES_PER_CHIRP)
w_dopp  = np.hanning(CHIRPS_PER_CPI)

freq = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1/FS)
freq = freq[:RANGE_BIN_LIMIT]

range_m = freq / HZ_PER_M
range_mask = range_m <= MAX_RANGE_TO_SHOW

range_m = range_m[range_mask]

doppler = np.fft.fftshift(np.fft.fftfreq(CHIRPS_PER_CPI, d=1/PRF))
velocity = doppler * lam / 2

# -----------------------------
# Setup plot
# -----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(10,6))

dummy = np.zeros((CHIRPS_PER_CPI, len(range_m)))

im = ax.imshow(dummy,
               aspect='auto',
               origin='lower',
               extent=[range_m[0], range_m[-1], velocity[0], velocity[-1]],
               vmin=-60, vmax=0)

ax.set_xlabel("Range (m)")
ax.set_ylabel("Velocity (m/s)")
title = ax.set_title("Range-Doppler Map")

fig.colorbar(im, ax=ax)
fig.tight_layout()

# -----------------------------
# MAIN LOOP (VIDEO)
# -----------------------------
# choose one CPI
from scipy.signal import hilbert

for cpi_idx in range(0, num_cpis, DISPLAY_STEP_CPI):

    cpi = cpis[cpi_idx]

    # convert to analytic signal (IQ)
    cpi = hilbert(cpi, axis=1)

    # remove DC per chirp
    cpi = cpi - np.mean(cpi, axis=1, keepdims=True)

    # RANGE FFT (complex now)
    rfft = np.fft.fft(cpi * w_range, axis=1)

    rfft[:, :5] = 0

    rfft = rfft[:, :RANGE_BIN_LIMIT]
    rfft = rfft[:, range_mask]

    # clutter removal (important)
    rfft = rfft - np.mean(rfft, axis=0, keepdims=True)

    # DOPPLER FFT
    rdm = np.fft.fft(rfft * w_dopp[:, None], axis=0)
    rdm = np.fft.fftshift(rdm, axes=0)

    mag = 20*np.log10(np.abs(rdm) + 1e-12)
    mag -= np.max(mag)

    im.set_data(mag)
    title.set_text(f"CPI {cpi_idx}")

    plt.pause(PAUSE_TIME)

plt.ioff()
plt.show()