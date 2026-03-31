import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as interp
import time


def hilbert_rvp(x, fs, kr):
    y = np.fft.fft(x, axis=-1)
    y[:, :y.shape[1] // 2 + 1] = 0   # keep negative-frequency analytic form like your original
    f = np.linspace(-fs / 2, fs / 2, y.shape[1])
    y *= np.exp(-1j * np.pi * f**2 / kr)
    return np.fft.ifft(y, axis=-1)


# ============================================================
# FILE READ
# ============================================================

record_file = open("Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_41_58_parking_lot_sar.txt", "r")
line_counter = 0

data = str(record_file.readline()); line_counter += 1
RECORD_COUNTER = int(int(data[0:len(data) - 1]) * 100 / 100)
print("Record Counter:", RECORD_COUNTER)

data = str(record_file.readline()); line_counter += 1
RECORD_TIME = int(data[0:len(data) - 1])
print("Record Time:", RECORD_TIME, "sec.")

data = str(record_file.readline()); line_counter += 1
SWEEP_TIME = int(data[0:len(data) - 1]) / 1e6
print("Sweep Time:", SWEEP_TIME, "sec.")

data = str(record_file.readline()); line_counter += 1
SWEEP_DELAY = int(data[0:len(data) - 1]) / 1e6
print("Sweep Delay:", SWEEP_DELAY, "sec.")

data = str(record_file.readline()); line_counter += 1
SWEEP_START = int(data[0:len(data) - 1])
print("Sweep Start:", SWEEP_START, "Hz")

data = str(record_file.readline()); line_counter += 1
SWEEP_BW = int(data[0:len(data) - 1])
print("Sweep BW:", SWEEP_BW, "Hz")

data = str(record_file.readline()); line_counter += 1
SAMPLING_FREQUENCY = int(data[0:len(data) - 1])
print("Sampling Frequency:", SAMPLING_FREQUENCY, "Hz")

data = str(record_file.readline()); line_counter += 1
NUMBER_OF_SAMPLES = int(data[0:len(data) - 1])
print("Samples per sweep:", NUMBER_OF_SAMPLES)

data = str(record_file.readline()); line_counter += 1
TX_MODE = int(data[0:len(data) - 1])
print("Tx Mode:", TX_MODE)

data = str(record_file.readline()); line_counter += 1
TX_POWER_DBM = int(data[0:len(data) - 1])
print("Tx Power:", TX_POWER_DBM, "dBm")

data = str(record_file.readline()); line_counter += 1
TX_POWER_DBM_VOLTAGE = int(data[0:len(data) - 1])
print("Tx Power Voltage:", TX_POWER_DBM_VOLTAGE / 100.0, "V")

data = str(record_file.readline()); line_counter += 1
hz_per_m = int(data[0:len(data) - 1])
print("Hz per m:", hz_per_m)

data = str(record_file.readline()); line_counter += 1
DATA_LOG = int(data[0:len(data) - 1])
print("Data Log:", DATA_LOG)

data = str(record_file.readline()); line_counter += 1
ADC_SELECT = int(data[0:len(data) - 1])
print("ADC Select:", ADC_SELECT)

data = str(record_file.readline()); line_counter += 1
USB_DATA_TYPE = int(data[0:len(data) - 1])
print("USB Data Type:", USB_DATA_TYPE)

data = str(record_file.readline()); line_counter += 1
ADC_RESOLUTION = int(data[0:len(data) - 1])
print("ADC Resolution:", ADC_RESOLUTION)

data = str(record_file.readline()); line_counter += 1
PHASE_DISTANCE = int(data[0:len(data) - 1])
print("Phase Distance:", PHASE_DISTANCE)

RECORD_DATE = str(record_file.readline()); line_counter += 1
print("Date:", RECORD_DATE)

# ============================================================
# USER PARAMETERS
# ============================================================

speed = 2.0                 # tune this
sample_increment = 1
data_counter = 0

dynamic_range = 50
cross_range_padding = 2      # 1 raw, 2-4 smoother
ky_delta_spacing = 1.5      # smaller than 1.8 -> better focus usually
window_fast = np.hanning
window_slow = np.hanning

remove_first_sweeps = 0
remove_leakage_bins = 10
remove_static_clutter = True

use_rvp = True               # keep True for Omega-K first
interp_kind = "cubic"        # 'linear' or 'cubic'

c = 299792458.0

# ============================================================
# LOAD RAW SWEEPS
# ============================================================

data1 = np.zeros([int(RECORD_COUNTER / sample_increment) - data_counter, NUMBER_OF_SAMPLES], dtype=np.float64)

for _ in range(data_counter):
    _ = record_file.readline()

sample_counter = 0

while data_counter < RECORD_COUNTER:
    sample_line = record_file.readline()
    samples_hex = bytes.fromhex(sample_line)
    length_line = len(samples_hex)

    if USB_DATA_TYPE == 0:
        samples_hex_ = [i / 150.0 for i in samples_hex]
        samples_float = [i * 3.3 for i in samples_hex_]
        data1[sample_counter, :] = samples_float
        sample_counter += 1

    elif USB_DATA_TYPE == 1:
        index = 0
        append_counter = 0

        while index < length_line:
            current_sample_16bit = ((samples_hex[index] & 0xFF) << 8) | (samples_hex[index + 1] & 0xFF)
            current_sample_float = (current_sample_16bit / (2 ** ADC_RESOLUTION)) * 3.3

            index += 2
            data1[sample_counter, append_counter] = current_sample_float
            append_counter += 1

        sample_counter += 1

    data_counter += sample_increment

record_file.close()

if remove_first_sweeps > 0:
    data1 = data1[remove_first_sweeps:, :]

start_time = time.time()

# ============================================================
# BASIC PARAMETERS
# ============================================================

data = np.array(data1, dtype=np.float64)

fs = SAMPLING_FREQUENCY
tsweep = SWEEP_TIME
tdelay = SWEEP_DELAY + ((SWEEP_TIME + SWEEP_DELAY) * (sample_increment - 1))
bw = SWEEP_BW
fc = SWEEP_START + bw / 2.0
sweep_samples = data.shape[1]

pri = tsweep + tdelay
delta_crange = pri * speed
lam = c / fc

print("Cross-range delta {:.6f} m, {:.6f} lambda".format(delta_crange, delta_crange / lam))
print("PRI {:.6f} s, PRF {:.2f} Hz".format(pri, 1.0 / pri))
print("Carrier {:.3f} GHz".format(fc / 1e9))

# ============================================================
# PREPROCESSING
# ============================================================

# Remove DC per chirp
data = data - np.mean(data, axis=1, keepdims=True)

# Fast-time window
wf = window_fast(sweep_samples)
data = data * wf

# Analytic signal + RVP compensation (for Omega-K path)
if use_rvp:
    data = hilbert_rvp(data, fs, bw / tsweep)

# Remove static clutter across chirps
if remove_static_clutter:
    data = data - np.mean(data, axis=0, keepdims=True)

# Optional quick debug
if 0:
    tmp = 20 * np.log10(np.abs(np.fft.fft(data, axis=1)) + 1e-12)
    plt.figure()
    plt.title("Range FFT debug")
    plt.imshow(tmp, aspect='auto', interpolation='none')
    plt.show()

# ============================================================
# CROSS-RANGE ZERO PADDING
# ============================================================

if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1) * data.shape[0])
    data = np.pad(data, ((zpad // 2, zpad // 2), (0, 0)), mode='constant')

# Azimuth window before along-track FFT
ws = window_slow(data.shape[0])[:, None]
data = data * ws

# ============================================================
# OMEGA-K SETUP
# ============================================================

# along-track spatial frequency
kx = np.linspace(-np.pi / delta_crange, np.pi / delta_crange, data.shape[0])

# RF wavenumber axis across sweep bandwidth
kr = np.linspace((4.0 * np.pi / c) * (fc - bw / 2.0),
                 (4.0 * np.pi / c) * (fc + bw / 2.0),
                 sweep_samples)

# Along-track FFT
cfft = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

# Optional leakage suppression near zero-range-like region
if remove_leakage_bins > 0:
    cfft[:, :remove_leakage_bins] = 0

# ============================================================
# STOLT INTERPOLATION
# ============================================================

kr_delta = kr[1] - kr[0]

# valid starting ky from worst-case kx
kx_max = np.max(np.abs(kx))
ky0 = np.sqrt(max(kr[0] ** 2 - kx_max ** 2, 1e-12))
ky_delta = ky_delta_spacing * kr_delta
ky_even = np.arange(ky0, kr[-1], ky_delta)

st = np.zeros((cfft.shape[0], len(ky_even)), dtype=np.complex128)

print("Entering Stolt interpolation")

for i in range(len(kx)):
    val = kr**2 - kx[i]**2
    valid = val > 0

    if np.count_nonzero(valid) < 4:
        continue

    ky = np.sqrt(val[valid])
    row = cfft[i, valid]

    # keep monotonic unique ky
    ky_unique, unique_idx = np.unique(ky, return_index=True)
    row_unique = row[unique_idx]

    if len(ky_unique) < 4:
        continue

    ci = interp.interp1d(
        ky_unique,
        row_unique,
        kind=interp_kind,
        bounds_error=False,
        fill_value=0.0
    )

    st[i, :] = ci(ky_even)

print("Finished Stolt interpolation")

# Optional 2D window after remap
if 1:
    wx = np.hanning(st.shape[0])
    wy = np.hanning(st.shape[1])
    st = st * np.sqrt(np.outer(wx, wy))

# ============================================================
# IMAGE FORMATION
# ============================================================

# optional smooth upsampling in image domain
if cross_range_padding > 1:
    pass

img = np.fft.ifft2(st)
img = np.fft.fftshift(img, axes=0)

# split halves
mid = img.shape[0] // 2
upper = img[:mid, :]   # currently top (far part)
lower = img[mid:, :]   # currently bottom (near part)

# reorder: near first, then far
img_combined = np.vstack((lower, upper))

# display extents
range1 = c * (fs / 2.0) * tsweep / (2.0 * bw)
crange = delta_crange * img_combined.shape[0]

# convert to dB
img_db = 20 * np.log10(np.abs(img_combined) + 1e-12)

# plot
plt.figure(figsize=(9,7))
plt.title("SAR Image (Combined Correct Order)")
plt.imshow(
    img_db,
    aspect='auto',
    interpolation='none',
    extent=[0, range1, -crange/2.0, crange/2.0],
    origin='lower'
)
m = np.max(img_db)
plt.clim(m - dynamic_range, m)
plt.colorbar(label='dB')
plt.xlabel("Range [m]")
plt.ylabel("Cross-range [m]")
plt.tight_layout()
plt.show()

# ============================================================
# OPTIONAL DEBUG PLOTS
# ============================================================

if 0:
    plt.figure()
    plt.title("Along-track FFT magnitude")
    plt.imshow(np.abs(cfft), aspect='auto', interpolation='none',
               extent=[kr[0], kr[-1], kx[0], kx[-1]])
    plt.xlabel("kr")
    plt.ylabel("kx")
    plt.show()

if 0:
    plt.figure()
    plt.title("Stolt magnitude")
    plt.imshow(np.abs(st), aspect='auto', interpolation='none',
               extent=[ky_even[0], ky_even[-1], kx[0], kx[-1]])
    plt.xlabel("ky")
    plt.ylabel("kx")
    plt.show()