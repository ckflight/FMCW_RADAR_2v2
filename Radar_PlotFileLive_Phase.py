import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.signal import lfilter

# -----------------------------
# User parameters
# -----------------------------
FIR_SMOOTHING_N   = 40     # moving-average taps (higher = smoother, slower response)
PLOT_WINDOW_SEC   = 4.0   # oscilloscope window length; set None to keep full history
USE_BLIT          = True   # faster redraw if backend supports it

# -----------------------------
# Open record file
# -----------------------------
record_file = open("Radar_Records/data_record.txt", "r")
# record_file = open("Radar_Records/radar2v2_horn_fc32k_600_phase_heartbeat_breath.txt", "r")

def read_int_line(f):
    return int(f.readline().strip())

# -----------------------------
# Read header (same as your format)
# -----------------------------
RECORD_COUNTER      = read_int_line(record_file)
print("Record Counter: ", RECORD_COUNTER)

RECORD_TIME         = read_int_line(record_file)
print("Record Time: ", RECORD_TIME, " sec.")

SWEEP_TIME          = read_int_line(record_file) / 1_000_000.0
print("Sweep Time : ", SWEEP_TIME, " microsec.")

SWEEP_DELAY         = read_int_line(record_file) / 1_000_000.0
print("Sweep Delay : ", SWEEP_DELAY, " microsec.")

SWEEP_START         = read_int_line(record_file)
print("Sweep Start : ", SWEEP_START, " Hz")

SWEEP_BW            = read_int_line(record_file)
print("Sweep BW : ", SWEEP_BW, " Hz")

SAMPLING_FREQUENCY  = read_int_line(record_file)
print("Sampling Frequency : ", SAMPLING_FREQUENCY, " Hz.")

NUMBER_OF_SAMPLES   = read_int_line(record_file)
print("Samples per sweep : ", NUMBER_OF_SAMPLES)

TX_MODE             = read_int_line(record_file)
print("Tx Mode : ", TX_MODE)

TX_POWER_DBM        = read_int_line(record_file)
print("Tx Power : ", TX_POWER_DBM, " dBm.")

TX_POWER_DBM_VOLTAGE = read_int_line(record_file)
print("Tx Power : ", TX_POWER_DBM_VOLTAGE / 100.0, " volts.")

hz_per_m            = read_int_line(record_file)
print("Hz per m : ", hz_per_m)

DATA_LOG            = read_int_line(record_file)
print("Data Log : ", DATA_LOG)

ADC_SELECT          = read_int_line(record_file)
print("ADC Select : ", ADC_SELECT)

USB_DATA_TYPE       = read_int_line(record_file)
print("USB Data Type : ", USB_DATA_TYPE)

ADC_RESOLUTION      = read_int_line(record_file)
print("ADC Resolution : ", ADC_RESOLUTION)

PHASE_DISTANCE      = read_int_line(record_file)
print("Phase Distance : ", PHASE_DISTANCE)

RECORD_DATE         = record_file.readline().rstrip("\n")
print("Date: ", RECORD_DATE)

# -----------------------------
# Derived constants (same meaning as your code)
# -----------------------------
distance_cm         = PHASE_DISTANCE
PHASE_FREQ          = (distance_cm / 100.0) * hz_per_m   # Hz
ST                  = SWEEP_TIME + SWEEP_DELAY           # seconds per sweep sample

# wavelength in mm at center freq (your original formula)
wavelength_mm       = 3e11 / (SWEEP_START + (SWEEP_BW / 2.0))
degrees_per_mm      = 360.0 / (wavelength_mm / 2.0)

# FFT config
N = NUMBER_OF_SAMPLES
window = np.hamming(N).astype(np.float32)

# IMPORTANT: match your original f_step definition:
# you used fs = 0.5/sample_period => fs = Fs/2 (Nyquist), then f_step = fs/bins
nyquist = SAMPLING_FREQUENCY / 2.0
bins = (N // 2) + 1
f_step = nyquist / bins

target_bin = int(PHASE_FREQ / f_step)
target_bin = max(0, min(target_bin, bins - 1))
print(f"Target phase bin: {target_bin} (f_step={f_step:.3f} Hz, PHASE_FREQ={PHASE_FREQ:.3f} Hz)")

# -----------------------------
# Streaming FIR (moving average)
# -----------------------------
n_taps = int(FIR_SMOOTHING_N)
b = (np.ones(n_taps, dtype=np.float32) / n_taps)
a = np.array([1.0], dtype=np.float32)
zi = np.zeros(n_taps - 1, dtype=np.float32)

# -----------------------------
# Plot buffers (limit points for performance)
# -----------------------------
if PLOT_WINDOW_SEC is None:
    maxlen = None
else:
    maxlen = max(10, int(PLOT_WINDOW_SEC / ST) + 1)

x_buf = deque(maxlen=maxlen)
y_buf = deque(maxlen=maxlen)

# -----------------------------
# Matplotlib setup
# -----------------------------
fig, ax = plt.subplots(figsize=(13, 7))
line, = ax.plot([], [], color="cyan", linewidth=1.5)

ax.set_xlabel("Time in sec")
ax.set_ylabel("Motion in micro meter")

ax.set_ylim(-50000, 50000)  # keep stable
ax.grid(True, alpha=0.4)

# If you want very dense ticks like your original, it hurts performance.
# Keep ticks reasonable for live plotting:
ax.set_xlim(0, max(1.0, PLOT_WINDOW_SEC if PLOT_WINDOW_SEC else RECORD_TIME))

# -----------------------------
# Fast per-frame processing: parse -> rfft -> phase(bin) -> displacement -> streaming FIR -> plot
# -----------------------------
data_counter = 0
time_counter = 0

def parse_samples_from_line(sample_line: str) -> np.ndarray:
    """
    Returns float32 voltage samples of length N.
    USB_DATA_TYPE 0: bytes 0..255 scaled roughly like your code ( /150 * 3.3 )
    USB_DATA_TYPE 1: big-endian uint16 scaled to 0..3.3V using ADC_RESOLUTION
    """
    # Convert hex string -> bytes
    raw = bytes.fromhex(sample_line.strip())

    if USB_DATA_TYPE == 0:
        u8 = np.frombuffer(raw, dtype=np.uint8)
        # Your original: samples_hex_ = [i/150] then *3.3
        v = (u8.astype(np.float32) * (3.3 / 150.0))
        return v

    # USB_DATA_TYPE == 1
    u16 = np.frombuffer(raw, dtype=">u2")  # big-endian unsigned 16-bit
    scale = 3.3 / (2 ** ADC_RESOLUTION)
    v = u16.astype(np.float32) * scale
    return v

def phase_at_target_bin(samples_v: np.ndarray) -> float:
    """
    Applies Hamming window, computes rFFT, returns phase (deg) at target_bin.
    """
    # Ensure expected length (robustness)
    if samples_v.size != N:
        # If your file format always matches N, you should never hit this.
        # If it happens, pad/trim to keep processing alive.
        if samples_v.size < N:
            samples_v = np.pad(samples_v, (0, N - samples_v.size), mode="constant")
        else:
            samples_v = samples_v[:N]

    xw = samples_v * window
    X = np.fft.rfft(xw)  # complex spectrum
    ph_deg = np.degrees(np.angle(X[target_bin]))
    return float(ph_deg)

def update(_frame):
    global data_counter, time_counter, zi

    if data_counter >= RECORD_COUNTER:
        return (line,)

    sample_line = record_file.readline()
    if not sample_line:
        return (line,)

    samples_v = parse_samples_from_line(sample_line)
    ph = phase_at_target_bin(samples_v)

    # Match your original mapping
    current_phase = ph + 180.0  # shift to 0..360-ish
    motion_um = (current_phase / degrees_per_mm) * 1000.0

    # time
    time_counter += 1
    t = time_counter * ST

    # streaming FIR
    y_filt, zi = lfilter(b, a, [motion_um], zi=zi)
    y_filt = float(y_filt[0])

    x_buf.append(t)
    y_buf.append(y_filt)

    # Update plot data
    # Converting deque->list each frame is OK because buffer length is bounded (windowed).
    line.set_data(list(x_buf), list(y_buf))

    # Sliding x-axis
    if PLOT_WINDOW_SEC is None:
        ax.set_xlim(0, max(1.0, t))
    else:
        left = max(0.0, t - PLOT_WINDOW_SEC)
        ax.set_xlim(left, left + PLOT_WINDOW_SEC)

    data_counter += 1
    return (line,)

ani = FuncAnimation(
    fig,
    update,
    interval=max(1, int(ST * 1000.0)),  # ms
    blit=USE_BLIT
)

plt.show()
