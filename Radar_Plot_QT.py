# pip3 install pyserial

import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as mp
import peakutils
import time
from scipy import signal
from scipy.signal import lfilter

from PyQt5 import QtWidgets, QtCore
from vispy.scene import SceneCanvas, visuals
from vispy.app import use_app
from vispy import scene

CANVAS_SIZE = (1400, 750)  # (width, height)

# Plotting 4 figures is slow so i will plot time and fft continuously.
# However other plots like phase and dbFs will be updated when they are selected.
UPDATE_FFT_PLOT     = 1
UPDATE_DBFS_PLOT    = 1
UPDATE_TIME_PLOT    = 1
UPDATE_PHASE_PLOT   = 0

USB_PLOT            = 1
MICROCARD_PLOT      = 0

USE_AVERAGE_FILTER  = 0
USE_FIR_FILTER      = 1
REMOVE_CLUTTER      = 0
FREQ_DIV            = 1
FIR_SMOOTHING_N     = 100 # higher is smoother

record_file = open("Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_28_06_parking_lot_run.txt", "r")
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
NUMBER_OF_SAMPLES_ORG = NUMBER_OF_SAMPLES
NUMBER_OF_SAMPLES -= 0
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

NUMBER_OF_INFO_LINES = line_counter
line_number_textFile = 0

is_restart_clicked = False

MAX_FREQ_RANGE      = SAMPLING_FREQUENCY / 2  # half of the maxsampling fre q.
FREQ_RANGE          = int(MAX_FREQ_RANGE / FREQ_DIV)
MAX_DISTANCE        = FREQ_RANGE / hz_per_m

if USE_AVERAGE_FILTER == 1:
    AVERAGING_NUM = 40
    MOVING_AVERAGE = AVERAGING_NUM / FREQ_DIV
    MOVING_AVERAGING_NUM = int(MOVING_AVERAGE / FREQ_DIV)  # higher averaging for higher fft range
    FFT_NUM_LINE_POINTS = int(
        (FREQ_RANGE / 2)) + 2 - MOVING_AVERAGING_NUM  # this is the array size according to moving averaging.

elif USE_FIR_FILTER == 1:
    n = FIR_SMOOTHING_N
    b = [1.0 / n] * n
    a = 1
    FFT_NUM_LINE_POINTS = int((FREQ_RANGE / 2))  # this is the array size according to moving averaging.

FFT_MAGNITUDE_FREQ_RANGE_MIN = 0.0
FFT_MAGNITUDE_FREQ_RANGE_MAX = float(FREQ_RANGE)

if REMOVE_CLUTTER:
    FFT_MAGNITUDE_MIN = 0
else:
    FFT_MAGNITUDE_MIN = 100
FFT_MAGNITUDE_MAX = 800

FFT_PHASE_FREQ_RANGE_MIN = 0
FFT_PHASE_FREQ_RANGE_MAX = float(FREQ_RANGE)

FFT_PHASE_ANGLE_MIN = -180
FFT_PHASE_ANGLE_MAX = 180

FFT_DBFS_FREQ_RANGE_MIN = 0
FFT_DBFS_FREQ_RANGE_MAX = float(FREQ_RANGE)
FFT_DBFS_MIN = -100
FFT_DBFS_MAX = 0

# time plot variables
TIME_MS = 50
TIME_NUM_LINE_POINTS = NUMBER_OF_SAMPLES * TIME_MS  # TIME_MS millisecond range plot

TIME_X_LIMIT_MIN = 0.0
TIME_X_LIMIT_MAX = float(TIME_MS / 1000)

TIME_Y_LIMIT_MIN = 0.0
TIME_Y_LIMIT_MAX = 3.3

# calculate this so that if record is 10 sec plot should take 10 seconds.
# FREQ_TIME_BETWEEN_PLOT = (FREQ_RANGE / SAMPLING_FREQUENCY) * 2 # Full freq range takes 1 second, half range 0.5 sec
# FREQ_TIME_BETWEEN_PLOT = TIME_BETWEEN_PLOT * 1  # i add this line to change plot speed. multp for slower div for faster

TIME_COLOR_CHOICES = ["black", "red", "blue"]
TIME_SCALE_CHOICES = ["scale/1", "scale/2", "scale/4", "scale/8", "scale/16", "scale/32", "scale/64", "scale/128"]

FREQ_COLOR_CHOICES = ["black", "red", "blue"]
FREQ_SCALE_CHOICES = ["scale/1", "scale/2", "scale/4", "scale/8", "scale/16", "scale/32", "scale/64", "scale/128"]

# Decode data_record
freq_sample_array_float = []  # scaled 3.3
freq_sample_array_int = []  # 12 bit
time_sample_array = []

clutter_counter = 0
fft_prev_array = np.zeros(FFT_NUM_LINE_POINTS)
fft_current_array = np.zeros(FFT_NUM_LINE_POINTS)


def Moving_Average_Filter(a, n=1):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def FFT_Calculate(sample_data_float, sample_data_int, sample_period):
    # This part is for fpga fir filter based radar
    '''
    # dbFs calculation for fir filter with gain.
    kaiser_beta = 8.6
    w = np.kaiser(len(sample_data_int), kaiser_beta)
    w *= len(w) / np.sum(w)

    fir_gain = 1
    x1 = sample_data_int
    x1 *= w / (fir_gain * 2 ** (ADC_RESOLUTION - 1))
    fx1 = 2*np.fft.rfft(x1)/(len(x1))
    fft_dbFs = 20*np.log10(np.abs(fx1))
    '''

    # FFt Calculation
    w = np.hamming(len(sample_data_float))
    sample_data1 = [sample_data_float[i] * w[i] for i in range(len(w))]

    fs = 0.5 / float(sample_period)

    # Calculated result of fft data is complex
    # Abs value gives magnitude and angle computation gives phase for each freq value.
    fft_ = np.fft.rfft(sample_data1)

    fft_abs = np.abs(fft_)
    fft_phs = np.angle(fft_, deg=True)

    # dbFs calculation
    fft_dbFs = 20 * np.log10((2 * fft_abs) / len(fft_abs))

    if USE_AVERAGE_FILTER == 1:
        fft_abs = Moving_Average_Filter(fft_abs, MOVING_AVERAGING_NUM)
        fft_phs = Moving_Average_Filter(fft_phs, MOVING_AVERAGING_NUM)
        fft_dbFs = Moving_Average_Filter(fft_dbFs, MOVING_AVERAGING_NUM)

    elif USE_FIR_FILTER == 1:
        fft_abs = lfilter(b, a, fft_abs)
        fft_phs = lfilter(b, a, fft_phs)
        fft_dbFs = lfilter(b, a, fft_dbFs)

        fft_abs = fft_abs[0:len(fft_abs) - 1]
        fft_phs = fft_phs[0:len(fft_phs) - 1]
        fft_dbFs = fft_dbFs[0:len(fft_dbFs) - 1]

    bins = len(fft_abs)
    f_step = fs / bins
    fx = [f_step * i for i in range(0, bins)]

    return fx, fft_abs, fft_phs, fft_dbFs


class Controls(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout()

        self.time_color_label = QtWidgets.QLabel("Time Color:")
        layout.addWidget(self.time_color_label)
        self.time_color_chooser = QtWidgets.QComboBox()
        self.time_color_chooser.addItems(TIME_COLOR_CHOICES)
        layout.addWidget(self.time_color_chooser)

        self.freq_color_label = QtWidgets.QLabel("Freq Color:")
        layout.addWidget(self.freq_color_label)
        self.freq_color_chooser = QtWidgets.QComboBox()
        self.freq_color_chooser.addItems(FREQ_COLOR_CHOICES)
        layout.addWidget(self.freq_color_chooser)

        self.time_scale_label = QtWidgets.QLabel("Time Scale:")
        layout.addWidget(self.time_scale_label)
        self.time_scale_chooser = QtWidgets.QComboBox()
        self.time_scale_chooser.addItems(TIME_SCALE_CHOICES)
        layout.addWidget(self.time_scale_chooser)

        self.freq_scale_label = QtWidgets.QLabel("Freq Scale:")
        layout.addWidget(self.freq_scale_label)
        self.freq_scale_chooser = QtWidgets.QComboBox()
        self.freq_scale_chooser.addItems(FREQ_SCALE_CHOICES)
        layout.addWidget(self.freq_scale_chooser)

        self.restart_label = QtWidgets.QLabel("Restart Plot")
        layout.addWidget(self.restart_label)
        self.restart_button = QtWidgets.QPushButton()
        self.restart_button.setText("Restart")
        layout.addWidget(self.restart_button)

        layout.addStretch(1)
        self.setLayout(layout)


class CanvasWrapper:
    def __init__(self):
        self.canvas = SceneCanvas(size=CANVAS_SIZE)
        self.grid = self.canvas.central_widget.add_grid()

        global line_number_textFile

        number_of_plot_row = 4
        view_counter = 0

        canvas_offset_per_row = CANVAS_SIZE[1] / number_of_plot_row

        # Time Plot
        # color array
        color = np.ones((TIME_NUM_LINE_POINTS, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, TIME_NUM_LINE_POINTS)
        color[:, 1] = color[::-1, 0]

        time_plot_data = np.zeros((TIME_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2x6000 array

        # background color
        self.view_time_plot = self.grid.add_view(view_counter, 0, bgcolor='#063970')
        self.time_line = visuals.Line(time_plot_data, parent=self.view_time_plot.scene, color=color)
        self.view_time_plot.camera = "panzoom"

        self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN, TIME_X_LIMIT_MAX),
                                             y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))

        # add x axis
        xaxis1 = scene.AxisWidget(orientation='top', axis_label='X Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        xaxis1.height_max = CANVAS_SIZE[1] - (view_counter * canvas_offset_per_row)

        self.grid.add_widget(xaxis1, row=0, col=0)
        xaxis1.link_view(self.view_time_plot)

        # add y axis
        yaxis1 = scene.AxisWidget(orientation='left', axis_label='Y Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        yaxis1.width_max = 0  # distance from right side of the window

        self.grid.add_widget(yaxis1, row=0, col=1)
        yaxis1.link_view(self.view_time_plot)
        view_counter += 1

        # Freq Magnitude Plot ***************************
        # color array
        color = np.ones((FFT_NUM_LINE_POINTS, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, FFT_NUM_LINE_POINTS)
        color[:, 1] = color[::-1, 0]

        fft_magnitude_plot_data = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2x6000 array

        # background color
        self.view_fft_magnitude_plot = self.grid.add_view(view_counter, 0, bgcolor='#063970')
        self.fft_magnitude_line = visuals.Line(fft_magnitude_plot_data, parent=self.view_fft_magnitude_plot.scene,
                                               color=color)
        self.view_fft_magnitude_plot.camera = "panzoom"

        self.view_fft_magnitude_plot.camera.set_range(x=(FFT_MAGNITUDE_FREQ_RANGE_MIN, FFT_MAGNITUDE_FREQ_RANGE_MAX),
                                                      y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))

        # add x axis
        xaxis2 = scene.AxisWidget(orientation='top', axis_label='X Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        xaxis2.height_max = CANVAS_SIZE[1] - (view_counter * canvas_offset_per_row)

        self.grid.add_widget(xaxis2, row=view_counter, col=0)
        xaxis2.link_view(self.view_fft_magnitude_plot)

        # add y axis
        yaxis2 = scene.AxisWidget(orientation='left', axis_label='Y Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        yaxis2.width_max = 0  # distance from right side of the window

        self.grid.add_widget(yaxis2, row=view_counter, col=1)
        yaxis2.link_view(self.view_fft_magnitude_plot)

        view_counter += 1

        # Freq Phase Plot ***************************
        # color array
        color = np.ones((FFT_NUM_LINE_POINTS, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, FFT_NUM_LINE_POINTS)
        color[:, 1] = color[::-1, 0]

        fft_phase_plot_data = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2x6000 array

        # background color
        self.view_fft_phase_plot = self.grid.add_view(view_counter, 0, bgcolor='#063970')
        self.fft_phase_line = visuals.Line(fft_phase_plot_data, parent=self.view_fft_phase_plot.scene, color=color)
        self.view_fft_phase_plot.camera = "panzoom"

        self.view_fft_phase_plot.camera.set_range(x=(FFT_PHASE_FREQ_RANGE_MIN, FFT_PHASE_FREQ_RANGE_MAX),
                                                  y=(FFT_PHASE_ANGLE_MIN, FFT_PHASE_ANGLE_MAX))

        # add x axis
        xaxis2 = scene.AxisWidget(orientation='top', axis_label='X Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        xaxis2.height_max = CANVAS_SIZE[1] - (view_counter * canvas_offset_per_row)

        self.grid.add_widget(xaxis2, row=view_counter, col=0)
        xaxis2.link_view(self.view_fft_phase_plot)

        # add y axis
        yaxis2 = scene.AxisWidget(orientation='left', axis_label='Y Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        yaxis2.width_max = 0  # distance from right side of the window

        self.grid.add_widget(yaxis2, row=view_counter, col=1)
        yaxis2.link_view(self.view_fft_phase_plot)
        view_counter += 1

        # Freq dbFS Plot ***************************
        # color array
        color = np.ones((FFT_NUM_LINE_POINTS, 4), dtype=np.float32)
        color[:, 0] = np.linspace(0, 1, FFT_NUM_LINE_POINTS)
        color[:, 1] = color[::-1, 0]

        fft_dbFs_plot_data = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2x6000 array

        # background color
        self.view_fft_dbFs_plot = self.grid.add_view(view_counter, 0, bgcolor='#063970')
        self.fft_dbFs_line = visuals.Line(fft_dbFs_plot_data, parent=self.view_fft_dbFs_plot.scene, color=color)
        self.view_fft_dbFs_plot.camera = "panzoom"

        self.view_fft_dbFs_plot.camera.set_range(x=(FFT_DBFS_FREQ_RANGE_MIN, FFT_DBFS_FREQ_RANGE_MAX),
                                                 y=(FFT_DBFS_MIN, FFT_DBFS_MAX))

        # add x axis
        xaxis2 = scene.AxisWidget(orientation='top', axis_label='X Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        xaxis2.height_max = CANVAS_SIZE[1] - (view_counter * canvas_offset_per_row)

        self.grid.add_widget(xaxis2, row=view_counter, col=0)
        xaxis2.link_view(self.view_fft_dbFs_plot)

        # add y axis
        yaxis2 = scene.AxisWidget(orientation='left', axis_label='Y Axis', axis_font_size=8, axis_label_margin=3,
                                  tick_label_margin=2)
        yaxis2.width_max = 0  # distance from right side of the window

        self.grid.add_widget(yaxis2, row=view_counter, col=1)
        yaxis2.link_view(self.view_fft_dbFs_plot)
        view_counter += 1

    def set_time_color(self, color):
        print(f"Changing line color to {color}")
        self.time_line.set_data(color=color)

    def set_freq_color(self, color):
        print(f"Changing line color to {color}")
        self.fft_magnitude_line.set_data(color=color)

    def set_time_scale(self, scale):
        if scale == "scale/1":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 1, TIME_X_LIMIT_MAX / 1),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        if scale == "scale/2":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 2, TIME_X_LIMIT_MAX / 2),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/4":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 4, TIME_X_LIMIT_MAX / 4),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/8":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 8, TIME_X_LIMIT_MAX / 8),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/16":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 16, TIME_X_LIMIT_MAX / 16),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/32":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 32, TIME_X_LIMIT_MAX / 32),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/64":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 64, TIME_X_LIMIT_MAX / 64),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))
        elif scale == "scale/128":
            self.view_time_plot.camera.set_range(x=(TIME_X_LIMIT_MIN / 128, TIME_X_LIMIT_MAX / 128),
                                                 y=(TIME_Y_LIMIT_MIN, TIME_Y_LIMIT_MAX))

    def set_freq_scale(self, scale):
        if scale == "scale/1":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 1, FFT_MAGNITUDE_FREQ_RANGE_MAX / 1),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        if scale == "scale/2":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 2, FFT_MAGNITUDE_FREQ_RANGE_MAX / 2),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/4":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 4, FFT_MAGNITUDE_FREQ_RANGE_MAX / 4),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/8":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 8, FFT_MAGNITUDE_FREQ_RANGE_MAX / 8),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/16":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 16, FFT_MAGNITUDE_FREQ_RANGE_MAX / 16),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/32":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 32, FFT_MAGNITUDE_FREQ_RANGE_MAX / 32),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/64":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 64, FFT_MAGNITUDE_FREQ_RANGE_MAX / 64),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))
        elif scale == "scale/128":
            self.view_fft_magnitude_plot.camera.set_range(
                x=(FFT_MAGNITUDE_FREQ_RANGE_MIN / 128, FFT_MAGNITUDE_FREQ_RANGE_MAX / 128),
                y=(FFT_MAGNITUDE_MIN, FFT_MAGNITUDE_MAX))

    def restart_plot(self):
        print("Restart is clicked")
        global is_restart_clicked
        is_restart_clicked = True

    def update_data(self, new_data_dict):
        # print("Updating data...")
        self.time_line.set_data(new_data_dict["time"])
        self.fft_magnitude_line.set_data(new_data_dict["freq_magnitude"])
        self.fft_phase_line.set_data(new_data_dict["freq_phase"])
        self.fft_dbFs_line.set_data(new_data_dict["freq_dbFs"])


def waterfall_data(size=512, phase_range=(-np.pi, np.pi), mag_range=(0, 10)):
    """Returns a complex array where X ramps phase and Y ramps magnitude."""
    p0, p1 = phase_range
    phase_ramp = np.linspace(p0, p1 - 1 / size, size)

    m0, m1 = mag_range
    mag_ramp = np.linspace(m1, m0 + 1 / size, size)

    phase_ramp, mag_ramp = np.meshgrid(phase_ramp, mag_ramp)

    return (mag_ramp * np.exp(1j * phase_ramp)).astype(np.complex64)


class MyMainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()

    def __init__(self, canvas_wrapper: CanvasWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFixedWidth(CANVAS_SIZE[0])
        self.setFixedHeight(CANVAS_SIZE[1])

        self.setWindowTitle("FMCW Radar by Cenk Keskin")

        central_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout()

        self._controls = Controls()
        main_layout.addWidget(self._controls)
        self._canvas_wrapper = canvas_wrapper
        main_layout.addWidget(self._canvas_wrapper.canvas.native)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self._connect_controls()

    def _connect_controls(self):
        self._controls.time_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_time_color)
        self._controls.freq_color_chooser.currentTextChanged.connect(self._canvas_wrapper.set_freq_color)
        self._controls.time_scale_chooser.currentTextChanged.connect(self._canvas_wrapper.set_time_scale)
        self._controls.freq_scale_chooser.currentTextChanged.connect(self._canvas_wrapper.set_freq_scale)
        self._controls.restart_button.clicked.connect(self._canvas_wrapper.restart_plot)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        self.close()
        return super().closeEvent(event)


class DataSource(QtCore.QObject):
    new_data = QtCore.pyqtSignal(dict)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._should_end = False

        # create arrays
        self._time_domain_data = np.zeros((TIME_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2 x NUM_LINE_POINTS array
        self._freq_domain_magnitude = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2 x NUM_LINE_POINTS array
        self._freq_domain_phase = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2 x NUM_LINE_POINTS array
        self._freq_domain_dbFs = np.zeros((FFT_NUM_LINE_POINTS, 2), dtype=np.float32)  # 2 x NUM_LINE_POINTS array

    def run_data_creation(self):

        print("Run data creation is starting")
        global line_number_textFile
        global record_file
        global is_restart_clicked
        global clutter_counter
        global fft_prev_array
        global fft_current_array

        while line_number_textFile < RECORD_COUNTER:

            if is_restart_clicked == True:

                print("Reset variables")

                is_restart_clicked = False

                # move file cursor to data lines, first NUMBER_OF_INFO_LINES lines are radar info data
                record_file.seek(0)

                line_number_textFile = 0

                # Read info lines
                for i in range(0, NUMBER_OF_INFO_LINES):
                    _line = record_file.readline()

                clutter_counter = 0

            if self._should_end:
                print("Data source is told to stop")
                break

            # if plot is fast add some time here

            sample_line = record_file.readline()
            samples_hex = bytes.fromhex(sample_line)  # get hex data from string
            length_line = len(samples_hex)

            index = 0

            # If there is data available
            if length_line > 0:

                while index < length_line:

                    if ADC_SELECT == 1:
                        # For faster operation i transfer 4 bit and 6 bit of the samples in 2 bytes
                        current_sample_16bit = ((samples_hex[index] & 0xF) << 6) | (samples_hex[index + 1] & 0x3F)
                        current_sample_float = (current_sample_16bit / 2 ** 10) * 5.0

                    elif ADC_SELECT == 0:

                        if USB_DATA_TYPE == 0:  # 8 bit data float scaled for higher usb data rate

                            current_sample_8bit = (samples_hex[index] & 0xFF)
                            current_sample_float = (current_sample_8bit / 150.0)
                            current_sample_float *= 3.3
                            current_sample_16bit = (int)(current_sample_float * (2 ** ADC_RESOLUTION))

                            index += 1
                            freq_sample_array_float.append(current_sample_float)
                            freq_sample_array_int.append(current_sample_16bit)
                            time_sample_array.append(current_sample_float)

                        # 16 bit raw adc data
                        elif USB_DATA_TYPE == 1:
                            current_sample_16bit = ((samples_hex[index] & 0xFF) << 8) | (samples_hex[index + 1] & 0xFF)
                            current_sample_float = (current_sample_16bit / 2 ** ADC_RESOLUTION) * 3.3

                            index += 2
                            freq_sample_array_float.append(current_sample_float)
                            freq_sample_array_int.append(current_sample_16bit)
                            time_sample_array.append(current_sample_float)

            # Add another plot and try plotting with pointwise update from left to right

            # Plot Freq Line
            if len(freq_sample_array_float) == FREQ_RANGE:

                fft_freq, fft_magnitude, fft_phase, fft_dbFs = FFT_Calculate(freq_sample_array_float,
                                                                             freq_sample_array_int,
                                                                             1 / SAMPLING_FREQUENCY)

                if UPDATE_PHASE_PLOT:
                    self._freq_domain_phase[:, 0] = fft_freq  # freq values from 0 to 1/2 samplig freq
                    self._freq_domain_phase[:, 1] = fft_phase  # amplitude values of each freq

                if UPDATE_DBFS_PLOT:
                    self._freq_domain_dbFs[:, 0] = fft_freq  # freq values from 0 to 1/2 samplig freq
                    self._freq_domain_dbFs[:, 1] = fft_dbFs  # amplitude values of each freq

                if REMOVE_CLUTTER == 1:

                    if clutter_counter == 0:
                        fft_prev_array = fft_magnitude
                        clutter_counter = clutter_counter + 1
                    else:
                        fft_current_array = fft_magnitude - fft_prev_array

                        # subtraction makes negative amplitude so i make them zero
                        fft_current_array = np.where(fft_current_array < 0, 0, fft_current_array)

                        fft_prev_array = fft_magnitude

                        clutter_counter = clutter_counter + 1

                    if clutter_counter > 0:

                        if UPDATE_FFT_PLOT:
                            self._freq_domain_magnitude[:, 0] = fft_freq  # freq values from 0 to 1/2 samplig freq
                            self._freq_domain_magnitude[:, 1] = fft_current_array  # amplitude values of each freq

                elif REMOVE_CLUTTER == 0:

                    if UPDATE_FFT_PLOT:
                        self._freq_domain_magnitude[:, 0] = fft_freq  # freq values from 0 to 1/2 samplig freq
                        self._freq_domain_magnitude[:, 1] = fft_magnitude  # amplitude values of each freq

                freq_sample_array_float.clear()
                freq_sample_array_int.clear()

            # Plot Time Line
            if len(time_sample_array) == TIME_NUM_LINE_POINTS:
                time_step = np.linspace(0.0, float(TIME_MS / 1000), len(time_sample_array))

                if UPDATE_TIME_PLOT:
                    self._time_domain_data[:, 0] = time_step
                    self._time_domain_data[:, 1] = time_sample_array

                time_sample_array.clear()

            line_number_textFile += 1

            data_dict = {
                "time": self._time_domain_data,
                "freq_magnitude": self._freq_domain_magnitude,
                "freq_phase": self._freq_domain_phase,
                "freq_dbFs": self._freq_domain_dbFs,
            }

            self.new_data.emit(data_dict)
            # print("Data creation")

            if line_number_textFile == RECORD_COUNTER:
                line_number_textFile = 0

                freq_sample_array_float.clear()
                freq_sample_array_int.clear()
                time_sample_array.clear()

        print("Data source finishing")
        self.finished.emit()

    def stop_data(self):
        print("Data source is quitting...")
        self._should_end = True


if __name__ == '__main__':
    app = use_app("pyqt5")
    app.create()

    canvas_wrapper = CanvasWrapper()
    win = MyMainWindow(canvas_wrapper)
    data_thread = QtCore.QThread(parent=win)
    data_source = DataSource()
    data_source.moveToThread(data_thread)

    # update the visualization when there is new data
    data_source.new_data.connect(canvas_wrapper.update_data)
    # start data generation when the thread is started
    data_thread.started.connect(data_source.run_data_creation)
    # if the data source finishes before the window is closed, kill the thread
    data_source.finished.connect(data_thread.quit, QtCore.Qt.DirectConnection)
    # if the window is closed, tell the data source to stop
    win.closing.connect(data_source.stop_data, QtCore.Qt.DirectConnection)
    # when the thread has ended, delete the data source from memory
    data_thread.finished.connect(data_source.deleteLater)

    win.show()
    data_thread.start()
    app.run()

    print("Waiting for data source to close gracefully...")
    data_thread.wait(5000)
