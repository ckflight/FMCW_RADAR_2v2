import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import lfilter

FIR_SMOOTHING_N     = 40 # higher is smoother

record_file = open("Radar_Records/radar2v2_horn_fc48k_600_phase_heartbeat_breath.txt", "r")
line_counter = 0

data = str(record_file.readline())
line_counter += 1
RECORD_COUNTER = int(data[0:len(data) - 1])
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

def FFT_Calculate(sample_data_float, sample_period):

    w = np.hamming(len(sample_data_float))
    sample_data1 = [sample_data_float[i] * w[i] for i in range(len(w))]

    fs = 0.5 / float(sample_period)
    fft_ = np.fft.rfft(sample_data1)

    fft_abs = np.abs(fft_)
    fft_phs = np.angle(fft_, deg=True)

    bins = len(fft_abs)
    f_step = fs/bins
    fx = [f_step*i for i in range(0, bins)]

    return fx[0:len(fx)-1], fft_phs[0:len(fft_abs)-1], f_step

distance                = PHASE_DISTANCE # in cm
PHASE_FREQ              = (distance / 100.0) * hz_per_m
ST                      = SWEEP_TIME + SWEEP_DELAY
wavelength              = 3e11 / (SWEEP_START + (SWEEP_BW /2)) # in mm
degrees_per_mm          = 360.0 / (wavelength / 2)

time_values = np.linspace(0, int(RECORD_TIME), int(RECORD_COUNTER)) # Record Counter points in time
samples_float_16bit = []

y = []
x = []
data_counter    = 0 # ignore start of the record
time_counter = 0

# Filter variables
n = FIR_SMOOTHING_N
b = [1.0 / n] * n
a = 1

# Read N samples and take their fft which will have N/2 fft abs values
# write them to fft_abs_2d's each column and plot waterfall.
while data_counter < RECORD_COUNTER:

    sample_line = record_file.readline()
    samples_hex = bytes.fromhex(sample_line)  # get hex data from string
    length_line = len(samples_hex)

    if USB_DATA_TYPE == 0:

        # at this point samples_hex array have float values of each adc samples
        samples_hex_ = [i / 150.0 for i in samples_hex]
        samples_float = [i * 3.3 for i in samples_hex_]

        fx, fp, fstep = FFT_Calculate(samples_float, 1 / SAMPLING_FREQUENCY)

    elif USB_DATA_TYPE == 1:

        index = 0
        while index < length_line:

            current_sample_16bit = ((samples_hex[index] & 0xFF) << 8) | (samples_hex[index + 1] & 0xFF)
            current_sample_float = (current_sample_16bit / 2 ** ADC_RESOLUTION) * 3.3

            index += 2
            samples_float_16bit.append(current_sample_float)

        if(len(samples_float_16bit) == 0):
            x = 0

        fx, fp, fstep = FFT_Calculate(samples_float_16bit, 1 / SAMPLING_FREQUENCY)
        samples_float_16bit.clear()

    # freq resolution is 1000 (fstep) so 12th element means 12k etc
    current_phase = (fp[int(PHASE_FREQ / fstep)])
    current_phase += 180.0 # 0 to 360

    time_counter += 1
    y.append((current_phase / degrees_per_mm)*1000) # in micro meters
    x.append(time_counter * ST) # in sec

    data_counter += 1

# This filter smooths data and gets rid of noise on it.
yy = lfilter(b, a, y)

fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(1, 1, 1)

ax.clear()

major_ticks_x = np.arange(0, 1000, 1)
minor_ticks_x = np.arange(0, 1000, 0.5)

major_ticks_y = np.arange(-50000, 50000, 1000)
minor_ticks_y = np.arange(-50000, 50000, 500)

ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.5)

ax.plot(x, yy, 'cyan')

plt.xlabel('Time in sec')
plt.ylabel('Motion in micro meter')
plt.show()
