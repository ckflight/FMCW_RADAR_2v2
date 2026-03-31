import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import interp1d
import time


def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c * f / (2 * (bw / sweep_length))


def hilbert_rvp(x, fs, kr):
    y = np.fft.fft(x, axis=-1)
    y[:, :y.shape[1] // 2 + 1] = 0  # Zero positive frequencies
    f = np.linspace(-fs / 2, fs / 2, y.shape[1])
    y *= np.exp(-1j * np.pi * f**2 / kr)
    return np.fft.ifft(y, axis=-1)


# ============================================================
# FILE READ
# ============================================================

record_file = open("Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_41_58_parking_lot_sar.txt", "r")
line_counter = 0

data = str(record_file.readline())
line_counter += 1
RECORD_COUNTER = int(int(data[0:len(data) - 1]) * 100 / 100)
print("Record Counter: ", str(RECORD_COUNTER))

data = str(record_file.readline())
line_counter += 1
RECORD_TIME = int(data[0:len(data) - 1])
print("Record Time: ", str(RECORD_TIME), " sec.")

data = str(record_file.readline())
line_counter += 1
SWEEP_TIME = int(data[0:len(data) - 1]) / 1000000
print("Sweep Time : ", str(SWEEP_TIME), " sec.")

data = str(record_file.readline())
line_counter += 1
SWEEP_DELAY = int(data[0:len(data) - 1]) / 1000000
print("Sweep Delay : ", str(SWEEP_DELAY), " sec.")

data = str(record_file.readline())
line_counter += 1
SWEEP_START = int(data[0:len(data) - 1])
print("Sweep Start : ", str(SWEEP_START), " Hz")

data = str(record_file.readline())
line_counter += 1
SWEEP_BW = int(data[0:len(data) - 1])
print("Sweep BW : ", str(SWEEP_BW), " Hz")

data = str(record_file.readline())
line_counter += 1
SAMPLING_FREQUENCY = int(data[0:len(data) - 1])
print("Sampling Frequency : ", str(SAMPLING_FREQUENCY), " Hz.")

data = str(record_file.readline())
line_counter += 1
NUMBER_OF_SAMPLES = int(data[0:len(data) - 1])
print("Samples per sweep : ", str(NUMBER_OF_SAMPLES))

data = str(record_file.readline())
line_counter += 1
TX_MODE = int(data[0:len(data) - 1])
print("Tx Mode : ", str(TX_MODE))

data = str(record_file.readline())
line_counter += 1
TX_POWER_DBM = int(data[0:len(data) - 1])
print("Tx Power : ", str(TX_POWER_DBM), " dBm.")

data = str(record_file.readline())
line_counter += 1
TX_POWER_DBM_VOLTAGE = int(data[0:len(data) - 1])
print("Tx Power Voltage : ", str(TX_POWER_DBM_VOLTAGE / 100.0), " volts.")

data = str(record_file.readline())
line_counter += 1
hz_per_m = int(data[0:len(data) - 1])
print("Hz per m : ", str(hz_per_m))

data = str(record_file.readline())
line_counter += 1
DATA_LOG = int(data[0:len(data) - 1])
print("Data Log : ", str(DATA_LOG))

data = str(record_file.readline())
line_counter += 1
ADC_SELECT = int(data[0:len(data) - 1])
print("ADC Select : ", str(ADC_SELECT))

data = str(record_file.readline())
line_counter += 1
USB_DATA_TYPE = int(data[0:len(data) - 1])
print("USB Data Type : ", str(USB_DATA_TYPE))

data = str(record_file.readline())
line_counter += 1
ADC_RESOLUTION = int(data[0:len(data) - 1])
print("ADC Resolution : ", str(ADC_RESOLUTION))

data = str(record_file.readline())
line_counter += 1
PHASE_DISTANCE = int(data[0:len(data) - 1])
print("Phase Distance : ", str(PHASE_DISTANCE))

RECORD_DATE = str(record_file.readline())
line_counter += 1
print("Date: ", str(RECORD_DATE))


# ============================================================
# USER PARAMETERS
# ============================================================

speed = 1.5                 # m/s, must match real platform speed
dynamic_range = 40          # dB for display
window = np.hanning
c = 299792458.0

sample_increment = 1
data_counter = 0

# image grid
img_x_min = -10.0           # cross-range min [m]
img_x_max = 10.0            # cross-range max [m]
img_y_min = 1.0             # range min [m]
img_y_max = 100.0           # range max [m]

img_x_points = 350
img_y_points = 500

# optional background removal
remove_mean_across_chirps = True

# phase sign: try +1 first, if image is worse, use -1
phase_sign = +1

# optional amplitude normalization per chirp
normalize_each_chirp = False


# ============================================================
# LOAD RAW SWEEPS
# ============================================================

data1 = np.zeros([int(RECORD_COUNTER / sample_increment) - data_counter, NUMBER_OF_SAMPLES])

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


# ============================================================
# BASIC PARAMETERS
# ============================================================

start_time = time.time()

data = np.array(data1, dtype=np.float64)

fs = SAMPLING_FREQUENCY
tsweep = SWEEP_TIME
tdelay = SWEEP_DELAY + ((SWEEP_TIME + SWEEP_DELAY) * (sample_increment - 1))
bw = SWEEP_BW
fc = SWEEP_START + bw / 2
sweep_samples = data.shape[1]
slope = bw / tsweep
pri = tsweep + tdelay
delta_crange = pri * speed
lam = c / fc

print('Cross range delta {:.6f} m, {:.6f} lambda'.format(delta_crange, delta_crange / lam))
print('Carrier frequency {:.3f} GHz'.format(fc / 1e9))
print('Wavelength {:.4f} m'.format(lam))
print('PRI {:.6f} s, PRF {:.2f} Hz'.format(pri, 1.0 / pri))

# theoretical max range from Nyquist
range1 = c * (fs / 2.0) * tsweep / (2 * bw)
print('Approx max unambiguous beat-limited range {:.2f} m'.format(range1))


# ============================================================
# PREPROCESSING
# ============================================================

# remove static DC/sample bias
data = data - np.mean(data, axis=1, keepdims=True)

# fast-time window
data = data * window(sweep_samples)

#********************************************************
# analytic signal + residual video phase compensation
#data = hilbert_rvp(data, fs, bw / tsweep)

# window
data = data * window(sweep_samples)

# remove DC per chirp
data = data - np.mean(data, axis=1, keepdims=True)

# RANGE FFT directly
range_profiles = np.fft.rfft(data, axis=1)
freq = np.fft.rfftfreq(sweep_samples, d=1/fs)
ranges = freq / hz_per_m

# remove dc /leakage here
range_profiles[:, 0:10] = 0

# remove static clutter across chirps
range_profiles = range_profiles - np.mean(range_profiles, axis=0, keepdims=True)

plt.figure()
for i in range(0, 50, 5):
    plt.plot(ranges[10:], 20*np.log10(np.abs(range_profiles[i, 10:]) + 1e-12))
plt.title("Multiple chirps")
plt.xlabel("Range [m]")
plt.ylabel("dB")
plt.grid()
plt.show()

#************************************************************

# optional clutter suppression across chirps
if remove_mean_across_chirps:
    data = data - np.mean(data, axis=0, keepdims=True)

# optional normalization
if normalize_each_chirp:
    chirp_norm = np.max(np.abs(data), axis=1, keepdims=True) + 1e-12
    data = data / chirp_norm


# ============================================================
# RANGE FFT
# ============================================================

range_profiles_full = np.fft.fft(data, axis=1)
freq_full = np.fft.fftfreq(sweep_samples, d=1.0 / fs)

# keep only nonnegative beat frequencies
pos_mask = freq_full >= 0
freq = freq_full[pos_mask]
range_profiles = range_profiles_full[:, pos_mask]

# convert beat frequency to range
ranges = c * freq / (2.0 * slope)

print("Range profiles shape:", range_profiles.shape)
print("Range axis min/max: {:.3f} m / {:.3f} m".format(ranges[0], ranges[-1]))


# ============================================================
# RADAR POSITIONS ALONG TRACK
# ============================================================

num_chirps = range_profiles.shape[0]
radar_x = np.arange(num_chirps, dtype=np.float64) * delta_crange
radar_x = radar_x - np.mean(radar_x)   # center aperture at x=0

print("Aperture length: {:.3f} m".format(radar_x[-1] - radar_x[0] if num_chirps > 1 else 0.0))
print("Number of chirps:", num_chirps)


# ============================================================
# IMAGE GRID
# ============================================================

x_img = np.linspace(img_x_min, img_x_max, img_x_points)
y_img = np.linspace(img_y_min, img_y_max, img_y_points)
X, Y = np.meshgrid(x_img, y_img, indexing='xy')

img = np.zeros_like(X, dtype=np.complex128)


plt.figure()
plt.title("Range FFT (first chirp)")
plt.plot(ranges, 20*np.log10(np.abs(range_profiles[0]) + 1e-12))
plt.xlabel("Range [m]")
plt.ylabel("dB")
plt.grid()
plt.show()

# ============================================================
# BACKPROJECTION
# ============================================================

bp_start = time.time()

for i in range(num_chirps):
    if i % 20 == 0:
        print(f"Backprojection chirp {i+1}/{num_chirps}")

    # exact slant range from radar position to every pixel
    R = np.sqrt((X - radar_x[i])**2 + Y**2)

    # interpolate real and imag separately
    interp_real = interp1d(
        ranges,
        np.real(range_profiles[i]),
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    interp_imag = interp1d(
        ranges,
        np.imag(range_profiles[i]),
        kind='linear',
        bounds_error=False,
        fill_value=0.0
    )

    s = interp_real(R) + 1j * interp_imag(R)

    # coherent phase correction
    img += s * np.exp(phase_sign * 1j * 4.0 * np.pi * R / lam)

bp_end = time.time()

print("Backprojection time:", bp_end - bp_start, "sec")


# ============================================================
# IMAGE DISPLAY
# ============================================================

img_mag = np.abs(img)
img_db = 20 * np.log10(img_mag + 1e-12)

m = np.max(img_db)

plt.figure(figsize=(9, 7))
plt.title('SAR Image - Backprojection')
plt.imshow(
    img_db,
    aspect='auto',
    interpolation='none',
    extent=[x_img[0], x_img[-1], y_img[0], y_img[-1]],
    origin='lower'
)
plt.xlabel('Cross-range [m]')
plt.ylabel('Range [m]')
plt.clim(m - dynamic_range, m)
plt.colorbar(label='dB')
plt.tight_layout()
plt.show()


# ============================================================
# OPTIONAL: ENTROPY
# ============================================================

img_sum = np.sum(img_mag) + 1e-12
entropy = -np.sum((img_mag / img_sum) * np.log((img_mag / img_sum) + 1e-12))
print('Entropy:', entropy)

end_time = time.time()
print("Total calculation time:", end_time - start_time, "sec")