import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate as interp
from scipy.signal import hilbert
import time

def lanczos_interp1d(x, y):

    a = 3
    def finterp(xn):
        y_new = np.zeros(xn.shape[0], dtype=y.dtype)
        diff = np.ediff1d(x, to_end=x[-1]-x[-2])
        for e, xi in enumerate(xn):
            if xi < x[0] or xi > x[-1]:
                continue
            x0 = np.searchsorted(x, xi)
            for i in range(max(0, x0-a), min(len(x), x0+a+1)):
                z = (xi - x[i]) / diff[i]
                y_new[e] += y[i] * np.sinc(z) * np.sinc(z/a)
        return y_new

    return finterp

def f_to_d(f, bw, sweep_length):
    c = 299792458.0
    return c*f/(2*(bw/sweep_length))

def r4_normalize(x, d, e=4):
    y = np.fft.rfft(x, axis=-1)
    n = d[-1]**e
    y = y*d**e/n
    return np.fft.irfft(y, axis=-1)

def rvp_compensation(x, f, kr):
    return x * np.exp(-1j * np.pi * f**2 / kr)

def hilbert_rvp(x, fs, kr):
    y = np.fft.fft(x, axis=-1)
    y[:,:y.shape[1]//2+1] = 0 # Zero positive frequencies
    # Residual video phase term compensation
    f = np.linspace(-fs/2, fs/2, y.shape[1])
    y *= np.exp(-1j * np.pi * f**2 / kr)
    return np.fft.ifft(y, axis=-1)


record_file = open("Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_41_58_parking_lot_sar.txt", "r")
line_counter = 0

data = str(record_file.readline())
line_counter += 1
RECORD_COUNTER = int(int(data[0:len(data) - 1]) * 70 / 100)
print("Record Counter: ", str(RECORD_COUNTER))

data = str(record_file.readline())
line_counter += 1
RECORD_TIME = int(data[0:len(data) - 1])
print("Record Time: ", str(RECORD_TIME), " sec.")

data = str(record_file.readline())
line_counter += 1
SWEEP_TIME = int(data[0:len(data) - 1]) / 1000000
print("Sweep Time : ", str(SWEEP_TIME), " microsec.")

data = str(record_file.readline())
line_counter += 1
SWEEP_DELAY = int(data[0:len(data) - 1]) / 1000000
print("Sweep Delay : ", str(SWEEP_DELAY), " microsec.")

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
print("Sampling Freqeuncy : ", str(SAMPLING_FREQUENCY), " Hz.")

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
print("Tx Power : ", str(TX_POWER_DBM_VOLTAGE / 100.0), " volts.")

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

rs                  = 0
speed               = 1.75
interpolate         = 1 #IFFT zero-padding amount, smooths final image
cross_range_padding = 2 #FFT zero-padding amount, increases cross-range with reduced resolution
dynamic_range       = 60 #Dynamic range of final image in dB
ky_delta_spacing    = 1.80
window              = np.hanning
c                   = 299792458

# rows of a column are adc data and columns are record counte
sample_increment    = 1
data_counter        = 0 # ignore first recordings until car moves

data1               = np.zeros([int(RECORD_COUNTER/sample_increment)-data_counter, NUMBER_OF_SAMPLES])

for i in range(data_counter):
    sample_line = record_file.readline()

sample_counter = 0

while data_counter < RECORD_COUNTER:

    sample_line = record_file.readline()
    samples_hex = bytes.fromhex(sample_line)  # get hex data from string
    length_line = len(samples_hex)

    if USB_DATA_TYPE == 0:
        # at this point samples_hex array have float values of each adc samples
        samples_hex_ = [i / 150.0 for i in samples_hex]
        samples_float = [i * 3.3 for i in samples_hex_]

        data1[sample_counter, :] = samples_float
        sample_counter += 1

    elif USB_DATA_TYPE == 1:

        index = 0
        append_counter = 0

        while index < length_line:

            current_sample_16bit = ((samples_hex[index] & 0xFF) << 8) | (samples_hex[index + 1] & 0xFF)
            current_sample_float = (current_sample_16bit / 2 ** ADC_RESOLUTION) * 3.3

            index += 2
            data1[sample_counter, append_counter] = current_sample_float
            append_counter += 1

        sample_counter += 1

    data_counter += sample_increment

start_time = time.time()

data = np.array(data1)

fs              = SAMPLING_FREQUENCY
tsweep          = SWEEP_TIME
tdelay          = SWEEP_DELAY + ((SWEEP_TIME + SWEEP_DELAY) * (sample_increment - 1))
bw              = SWEEP_BW
fc              = SWEEP_START + bw/2
sweep_samples   = len(data[0])
delta_crange    = (tsweep + tdelay) * speed
print('Cross range delta {:.3f} m, {:.3f} lambda'.format(delta_crange, delta_crange/(c/fc)))

f = np.linspace(0, fs/2, sweep_samples//2+1)
d = f_to_d(f, bw, tsweep)

range0      = 0
range1      = c*(fs/2.0)*tsweep/(2*bw)
delta_range = range1/sweep_samples
crange0     = -len(data)*delta_crange/2.0
crange1     = len(data)*delta_crange/2.0
raw_extent  = (range0, range1, crange0, crange1)

#Window data
data = data * window(sweep_samples)

print('Sweep points', sweep_samples)

#Hilbert transformation to get complex data
data = hilbert_rvp(data, fs, bw/tsweep)

# plot raw data
if 0:
    shdata = 20*np.log10(np.abs(np.real([np.fft.rfft(r) for r in data])))
    plt.figure()
    plt.title('Raw data, range FFT')
    imgplot = plt.imshow(shdata, aspect='auto', interpolation='none', extent=raw_extent)
    plt.xlabel('Range [m]')
    plt.ylabel('Cross-range [m]')
    m = np.max(shdata)
    #Limit the dynamic range to clean the rounding errors
    imgplot.set_clim(m-dynamic_range,m)

if 0:
    plt.figure()
    plt.title('Raw data')
    kr0 = (4*np.pi/c)*(fc - bw/2)
    kr1 = (4*np.pi/c)*(fc + bw/2)
    plt.imshow(data.real, aspect='auto', interpolation='none', extent=(kr0, kr1, crange0, crange1))
    plt.xlabel('Range wavenumber [1/m]')
    plt.ylabel('Cross-range [m]')

plt.show()

#Zeropad cross-range
if cross_range_padding > 1:
    zpad = int((cross_range_padding - 1)*data.shape[0])
    data = np.pad(data, ((zpad//2, zpad//2), (0, 0)), 'constant')

kx = np.linspace(-np.pi/delta_crange, np.pi/delta_crange, len(data))
kr = np.linspace(((4*np.pi/c)*(fc - bw/2)), ((4*np.pi/c)*(fc + bw/2)), sweep_samples);

#along the track fft
cfft = np.fft.fftshift(np.fft.fft(data, axis=0), 0)

if 0:
    plt.figure()
    plt.title('Along track FFT phase')
    plt.imshow(np.angle(cfft), aspect='auto', extent=[kr[0], kr[-1], kx[0], kx[-1]])
    plt.figure()
    plt.title('Along track FFT magnitude')
    plt.imshow(np.abs(cfft), aspect='auto', extent=[kr[0], kr[-1], kx[0], kx[-1]])

#matched filter
if rs != 0:
    phi_mf = np.zeros(cfft.shape)
    for ii in range(cfft.shape[1]):
        for jj in range(cfft.shape[0]):
            phi_mf = rs*(kr[ii]**2-kx[jj]**2 )**0.5

    smf = np.exp(1j*phi_mf)
    cfft  = cfft*smf

#ky0 = (-1*(kr[0]**2 - kx[0]**2))**0.5 # kr - kx is negative gives nan
ky0 = (kr[0]**2 - kx[0]**2)**0.5
kr_delta = kr[1] - kr[0]
ky_delta = ky_delta_spacing * kr_delta
ky_even = np.arange(ky0, kr[-1], ky_delta)

st = np.zeros((cfft.shape[0], len(ky_even)), dtype=np.complex_(1))

print("entering slot interpolation")
#Stolt interpolation
for i in range(len(kx)):
    ky = (kr**2 - kx[i]**2)**0.5
    ci = interp.interp1d(ky, cfft[i], fill_value=0, bounds_error=False)
    #ci = lanczos_interp1d(ky, cfft[i])
    st[i,:] = ci(ky_even)
print("finished slot interpolation")
if 0:
    plt.figure()
    plt.title('Stolt interpolation phase')
    plt.imshow(np.angle(st), aspect='auto', extent=[ky_even[0], ky_even[-1], kx[0], kx[-1]])
    plt.figure()
    plt.title('Stolt interpolation magnitude')
    plt.imshow(np.abs(st), aspect='auto', extent=[ky_even[0], ky_even[-1], kx[0], kx[-1]])

#Window again
#wx = window(st.shape[0])
#wy = window(st.shape[1])
#w = np.sqrt(np.outer(wx, wy))
#st = st * w

end_time = time.time()

print("calculation time:",end_time-start_time)

#IFFT to generate the image
st = np.fft.ifft2(st)

st_sum = np.sum(np.abs(st))
print('Entropy', -np.sum((np.abs(st)/st_sum) * np.log(np.abs(st)/st_sum)))

#Cross-range size of image in meters
crange = delta_crange*st.shape[0]/interpolate
max_range = range1 * ky_delta / (2 * kr_delta)

plt.figure()
plt.title('SAR Image')

st = 20*np.log10(np.abs(st))
imgplot = plt.imshow(st, aspect='auto', interpolation='none', extent=[0, range1,-crange/2.0,crange/2.0], origin='lower')
m = np.max(st)
#Limit the dynamic range to clean the rounding errors
imgplot.set_clim(m-dynamic_range,m)

plt.show()
