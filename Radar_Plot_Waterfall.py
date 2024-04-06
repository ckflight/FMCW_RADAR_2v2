import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import lfilter

record_file = open("Radar_Records/sar_2.txt", "r")
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
SWEEP_GAP = int(data[0:len(data) - 1]) / 1000000
print("Sweep Gap : ", str(SWEEP_GAP), " microsec.")

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

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)

    # The code uses normal array convert np array to normal array
    temp_array = []
    for i in range(np.size(y)):
        temp_array.append(y[i])

    return temp_array

def Moving_Average_Filter(a, n=1):

    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def FFT_Calculate(sample_data_float, sample_period):

    w = np.hamming(len(sample_data_float))
    sample_data1 = [sample_data_float[i] * w[i] for i in range(len(w))]

    fs = 0.5 / float(sample_period)
    fft_ = np.fft.rfft(sample_data1)

    fft_abs = np.abs(fft_)

    if USE_AVERAGE_FILTER == 1:
        fft_abs = Moving_Average_Filter(fft_abs, AVERAGING_NUM)

    elif USE_FIR_FILTER == 1:
        fft_abs = lfilter(b, a, fft_abs)

    fft_abs = fft_abs[0:len(fft_abs) - 1]

    bins = len(fft_abs)
    f_step = fs/bins
    fx = [f_step*i for i in range(0, bins)]

    return fx, fft_abs

USE_AVERAGE_FILTER  = 0
USE_FIR_FILTER      = 0
REMOVE_CLUTTER      = 0
HIGH_PASS_FILTER    = 0

HIGHPASS_CUTOFF     = 10000
HIGHPASS_ORDER      = 5

if USE_AVERAGE_FILTER == 1:
    AVERAGING_NUM = 20

elif USE_FIR_FILTER == 1:
    n = 4  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1

time_values = np.linspace(0, int(RECORD_TIME), int(RECORD_COUNTER)) # Record Counter points in time
# fft result fx is same for each calculation which is the freq range.
fft_abs_2d  = np.zeros([int(NUMBER_OF_SAMPLES/2), int(len(time_values))])
samples_float_16bit = []
fft_current_array = np.zeros(int(NUMBER_OF_SAMPLES/2))

min_fft_value           = 100000.0 # some number to start finding smaller value
max_fft_value           = 0.0
average_fft_value       = 0.0
average_value_counter   = 0
clutter_counter         = 0
data_counter            = 0

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

        fx, fy = FFT_Calculate(samples_float, 1 / SAMPLING_FREQUENCY)
        fx = [i / hz_per_m for i in fx]

    elif USB_DATA_TYPE == 1:

        index = 0
        append_counter = 0
        while index < length_line:

            current_sample_16bit = ((samples_hex[index] & 0xFF) << 8) | (samples_hex[index + 1] & 0xFF)
            current_sample_float = (current_sample_16bit / 2 ** ADC_RESOLUTION) * 3.3

            index += 2
            samples_float_16bit.append(current_sample_float)
            append_counter += 1


        if HIGH_PASS_FILTER == 1:
            samples_float_16bit = butter_highpass_filter(samples_float_16bit, HIGHPASS_CUTOFF, SAMPLING_FREQUENCY, HIGHPASS_ORDER)

        fx, fy = FFT_Calculate(samples_float_16bit, 1 / SAMPLING_FREQUENCY)
        fx = [i / hz_per_m for i in fx]
        samples_float_16bit.clear()

    if REMOVE_CLUTTER == 1:

        if clutter_counter == 0:
            fft_prev_array = fy
            clutter_counter = clutter_counter + 1
        else:
            fft_current_array = fy - fft_prev_array

            # subtraction makes negative amplitude so i make them zero
            fft_current_array = np.where(fft_current_array < 0, 0, fft_current_array)
            fft_prev_array = fy
            clutter_counter = clutter_counter + 1

        if clutter_counter > 0:

            max_val = np.max(fy)
            min_val = np.min(fy)
            average_fft_value += np.average(fy)
            average_value_counter += 1

            if max_val > max_fft_value:
                max_fft_value = max_val

            if min_val <= min_fft_value and min_fft_value != 0.0:
                min_fft_value = min_val

            # fx is 0 1k 2k .... 1.860.000 values with 1860 size array
            fft_abs_2d[:, data_counter] = fft_current_array
            data_counter += 1

    elif REMOVE_CLUTTER == 0:

        max_val = np.max(fy)
        min_val = np.min(fy)
        average_fft_value += np.average(fy)
        average_value_counter += 1

        if max_val > max_fft_value:
            max_fft_value = max_val

        if min_val <= min_fft_value and min_fft_value != 0.0:
            min_fft_value = min_val

        # fx is 0 1k 2k .... 1.860.000 values with 1860 size array
        fft_abs_2d[:,data_counter] = fy
        data_counter += 1

average_fft_value /= average_value_counter
print("Min fft value:", min_fft_value)
print("Max fft value:", max_fft_value)
print("Average fft value:", average_fft_value)
fig = plt.figure(figsize=(12, 6))
ax = fig.subplots()

vmax = 0.0
if HIGH_PASS_FILTER == 1:
    vmax = max_fft_value
else:
    vmax = average_fft_value * 1

# Colormap color options:
#https://matplotlib.org/stable/users/explain/colors/colormaps.html

# colormesh needs x and y arrays for determining the axis' ranges which are fix values
# 2d array of magnitude values for each (x,y) coordinates which is a dynamic array.
cmap = ax.pcolormesh(time_values, fx, fft_abs_2d,
                     vmin=min_fft_value, vmax=vmax,
                     cmap=plt.colormaps['plasma'])

print("cmap:",cmap)
fig.colorbar(cmap)

plt.show()