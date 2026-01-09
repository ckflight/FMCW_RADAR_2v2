"""
Combined Recorder + Live Plot (low-latency, high-performance)

Key performance upgrades vs your current approach:
1) Single-bin phase using Goertzel (no full FFT)
2) Low-latency plotting: always plot the LATEST frame (drop backlog)
3) Streaming FIR filter with lfilter(zi=...) (O(Ntap) per sample)
4) Bounded queue + bounded plot window
5) Throttled y-autoscale (optional)

Tested assumptions:
- USB_DATA_TYPE == 1 means big-endian uint16 samples (as your file shows)
- FRAME BYTES = NUMBER_OF_SAMPLES*2 for USB_DATA_TYPE==1
- Uses your same header file format (so your offline reader still works)

Requirements:
pip install numpy matplotlib scipy pyserial
"""

import os
import time
import binascii
import platform
import threading
import queue
from datetime import datetime
from collections import deque

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import lfilter

# ============================================================
# USER CONFIG (from your record code)
# ============================================================
MEASUREMENT_TYPE    = "terrace"
ANTENNA_TYPE        = "horn"
RECORD_TIME         = 160            # seconds
NAME_ENCODE         = 0             # 0 => Radar_Records/data_record.txt , 1 => timestamped name

SWEEP_START         = 5.20e9
SWEEP_BW            = 400e6
SWEEP_TIME          = 1.0e-3
SWEEP_DELAY         = 3.0e-3

ADC_FS              = 3720000
ADC_BITS            = 16
ADC_AVERAGING       = 4

PHASE_DISTANCE      = 285 + 110     # cm

USB_DATA_TYPE       = 1             # 0: 8-bit scaled, 1: 16-bit
DATA_LOG            = 0
SWEEP_TYPE          = 0
USE_PLL             = 1
TX_MODE             = 1
GAIN                = 4
CHECK_MODE          = 0
TEST_DEVICE         = 1             # 1: STM32H7 serial on Linux (/dev/ttyACM0)
SALLENKEY_FC_KHZ    = 48
ADC_SELECT          = 0

# ============================================================
# LIVE PLOT + PERFORMANCE CONFIG
# ============================================================
FIR_SMOOTHING_N     = 30            # moving average taps
PLOT_WINDOW_SEC     = 2.0           # smaller window => faster plot (try 3..10); None => full history (slower)

TARGET_UI_FPS       = 60            # UI refresh rate (matplotlib practical range 20..60)
USE_BLIT            = True          # may help; if stutter, set False

LOW_LATENCY_MODE    = True          # True => always plot newest frame (drop backlog)
QUEUE_MAX           = 5             # keep small to avoid lag buildup

AUTOSCALE_Y         = True
AUTOSCALE_EVERY     = 20            # update y-limits every N redraws (higher => faster)
Y_MARGIN            = 0.15
Y_MIN_SPAN          = 200.0         # um

# ============================================================
# DERIVED PARAMETERS (matching your original intent)
# ============================================================
if ADC_SELECT == 0:
    SAMPLING_FREQUENCY = int(ADC_FS / ADC_AVERAGING)
    NUMBER_OF_SAMPLES  = int(SAMPLING_FREQUENCY / 1000) * 1
else:
    SAMPLING_FREQUENCY = 400000
    NUMBER_OF_SAMPLES  = 400

# hz_per_m exactly as your record code (distance=1m)
hz_per_m = 0
if SWEEP_BW:
    hz_per_m = int((2 * SWEEP_BW * 1.0) / (299792458.0 * SWEEP_TIME))

ST          = SWEEP_TIME + SWEEP_DELAY
SWEEP_FREQ  = 1.0 / ST
RECORD_COUNTER = int(RECORD_TIME * SWEEP_FREQ)

# Phase target frequency from distance bin
PHASE_FREQ = (PHASE_DISTANCE / 100.0) * hz_per_m

# Your displacement scale (same as your plotter)
wavelength_mm  = 3e11 / (SWEEP_START + (SWEEP_BW / 2.0))
degrees_per_mm = 360.0 / (wavelength_mm / 2.0)

# ============================================================
# SERIAL + CONFIGURATION (mirrors your Configuration_Process)
# ============================================================
def serial_init_specific(port_name: str) -> serial.Serial:
    ser = serial.Serial(
        port=port_name,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS,
        timeout=None
    )
    ser.flushInput()
    return ser

def configuration_process(ser: serial.Serial):
    isDone = False
    state = 0

    while not isDone:
        if state == 0:
            ser.write(b"==")

            def send_u16(val):
                v = np.uint16(val)
                ser.write(binascii.hexlify(np.uint8((v >> 8) & 0xFF)))
                ser.write(binascii.hexlify(np.uint8(v & 0xFF)))

            def send_u8(val):
                ser.write(binascii.hexlify(np.uint8(val)))

            send_u16(SWEEP_TIME * 1e6)
            send_u16(SWEEP_DELAY * 1e6)
            send_u8(RECORD_TIME)

            send_u16(SAMPLING_FREQUENCY / 1e3)
            send_u16(NUMBER_OF_SAMPLES)

            send_u16(SWEEP_START / 1e7)
            send_u16(SWEEP_BW / 1e6)

            send_u8(TX_MODE)
            send_u8(GAIN)
            send_u8(SWEEP_TYPE)
            send_u8(DATA_LOG)
            send_u8(ADC_SELECT)
            send_u8(USE_PLL)
            send_u8(CHECK_MODE)
            send_u8(USB_DATA_TYPE)
            send_u8(ADC_BITS)
            send_u8(ADC_AVERAGING)

            state = 1

        elif state == 1:
            while ser.inWaiting() == 0 and state == 1:
                pass

            if ser.inWaiting() > 0:
                config_read = ser.read(ser.inWaiting())
                if config_read[0:2] == b"==":
                    dbm  = config_read[2]
                    dbmv = config_read[3]
                    ver_major = config_read[4]
                    ver_minor = config_read[5]
                    isDone = True

    return int(dbm), int(dbmv), int(ver_major), int(ver_minor)

# ============================================================
# FILE OUTPUT (same header format as your recorder)
# ============================================================
def make_record_filename(radar_ver_major, radar_ver_minor):
    record_time_now = str(datetime.now().replace(microsecond=0))
    y = record_time_now[0:4]
    mo = record_time_now[5:7]
    d = record_time_now[8:10]
    h = record_time_now[11:13]
    mi = record_time_now[14:16]
    s = record_time_now[17:19]

    record_string = (
        f"radar{radar_ver_major}v{radar_ver_minor}"
        f"_{ANTENNA_TYPE}_{SALLENKEY_FC_KHZ}kHz"
        f"_{y}_{mo}_{d}_{h}_{mi}_{s}"
        f"_{MEASUREMENT_TYPE}"
    )
    return record_string, record_time_now

def open_record_file(record_string):
    os.makedirs("Radar_Records", exist_ok=True)

    if NAME_ENCODE == 0:
        path = "Radar_Records/data_record.txt"
    else:
        path = f"Radar_Records/{record_string}.txt"

    if os.path.exists(path):
        os.remove(path)

    return open(path, "w"), path

def write_header(f, record_time_now, tx_power_dbm, tx_power_dbm_voltage):
    f.write(str(RECORD_COUNTER) + "\r\n")
    f.write(str(RECORD_TIME) + "\r\n")
    f.write(str(int(SWEEP_TIME * 1_000_000)) + "\r\n")
    f.write(str(int(SWEEP_DELAY * 1_000_000)) + "\r\n")
    f.write(str(int(SWEEP_START)) + "\r\n")
    f.write(str(int(SWEEP_BW)) + "\r\n")
    f.write(str(SAMPLING_FREQUENCY) + "\r\n")
    f.write(str(NUMBER_OF_SAMPLES) + "\r\n")
    f.write(str(TX_MODE) + "\r\n")
    f.write(str(tx_power_dbm) + "\r\n")
    f.write(str(tx_power_dbm_voltage) + "\r\n")
    f.write(str(hz_per_m) + "\r\n")
    f.write(str(DATA_LOG) + "\r\n")
    f.write(str(ADC_SELECT) + "\r\n")
    f.write(str(USB_DATA_TYPE) + "\r\n")
    f.write(str(ADC_BITS) + "\r\n")
    f.write(str(PHASE_DISTANCE) + "\r\n")
    f.write(str(record_time_now) + "\r\n")
    f.flush()

# ============================================================
# FAST SINGLE-BIN DFT (Goertzel)
# ============================================================
N = int(NUMBER_OF_SAMPLES)
window = np.hamming(N).astype(np.float32)

# Bin mapping consistent with your original approach:
# you used nyquist = Fs/2 and bins = N/2+1 and f_step = nyquist/bins
nyquist = SAMPLING_FREQUENCY / 2.0
bins = (N // 2) + 1
f_step = nyquist / bins
target_bin = int(PHASE_FREQ / f_step)
target_bin = max(0, min(target_bin, bins - 1))

# Goertzel uses k defined over N-point DFT.
# For rfft bin k, it corresponds to DFT bin k in N-point DFT.
k_goertzel = target_bin

def goertzel_bin(x: np.ndarray, k: int) -> complex:
    """
    Goertzel for DFT bin k of an N-point sequence x (N=len(x)).
    Returns complex X[k].
    """
    Nloc = x.size
    w = 2.0 * np.pi * k / Nloc
    cosw = np.cos(w)
    sinw = np.sin(w)
    coeff = 2.0 * cosw

    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    # Python loop; still usually faster than full FFT when only 1 bin is needed.
    for xn in x:
        s0 = float(xn) + coeff * s1 - s2
        s2 = s1
        s1 = s0

    real = s1 - cosw * s2
    imag = sinw * s2
    return real + 1j * imag

def bytes_to_volts(raw: bytes) -> np.ndarray:
    if USB_DATA_TYPE == 0:
        u8 = np.frombuffer(raw, dtype=np.uint8)
        return (u8.astype(np.float32) * (3.3 / 150.0))

    u16 = np.frombuffer(raw, dtype=">u2")  # big-endian uint16
    scale = 3.3 / (2 ** ADC_BITS)
    return u16.astype(np.float32) * scale

def phase_deg_single_bin(samples_v: np.ndarray) -> float:
    if samples_v.size != N:
        if samples_v.size < N:
            samples_v = np.pad(samples_v, (0, N - samples_v.size), mode="constant")
        else:
            samples_v = samples_v[:N]
    xw = samples_v * window
    Xk = goertzel_bin(xw, k_goertzel)
    return float(np.degrees(np.angle(Xk)))

# ============================================================
# STREAMING FIR (moving average)
# ============================================================
n_taps = int(FIR_SMOOTHING_N)
b = (np.ones(n_taps, dtype=np.float32) / n_taps)
a = np.array([1.0], dtype=np.float32)
zi = np.zeros(n_taps - 1, dtype=np.float32)

# ============================================================
# THREADING: ACQUISITION -> QUEUE
# ============================================================
stop_event = threading.Event()
frame_q = queue.Queue(maxsize=QUEUE_MAX)

def acquisition_loop(ser: serial.Serial, out_f):
    bytes_per_frame = NUMBER_OF_SAMPLES if USB_DATA_TYPE == 0 else (NUMBER_OF_SAMPLES * 2)

    rx_counter = 0
    t0 = time.time()

    while (not stop_event.is_set()) and (rx_counter < RECORD_COUNTER):
        raw = ser.read(bytes_per_frame)
        if not raw:
            continue

        # 1) Record to file exactly as you do
        hex_str = binascii.b2a_hex(raw).decode("ascii")
        out_f.write(hex_str + "\r\n")

        # 2) Push to queue
        if LOW_LATENCY_MODE:
            # keep newest; if full, drop one old and push new
            try:
                frame_q.put_nowait(raw)
            except queue.Full:
                try:
                    _ = frame_q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frame_q.put_nowait(raw)
                except queue.Full:
                    pass
        else:
            # keep all (may lag)
            frame_q.put(raw)

        rx_counter += 1

        if rx_counter % 100 == 0:
            dt = time.time() - t0
            print(f"rx={rx_counter}/{RECORD_COUNTER} ({rx_counter/dt:.1f} frames/s)")

    out_f.flush()
    stop_event.set()
    print("Acquisition done.")

# ============================================================
# LIVE PLOT SETUP
# ============================================================
if PLOT_WINDOW_SEC is None:
    maxlen = None
else:
    maxlen = max(10, int(PLOT_WINDOW_SEC / ST) + 1)

x_buf = deque(maxlen=maxlen)
y_buf = deque(maxlen=maxlen)

fig, ax = plt.subplots(figsize=(13, 7))
line, = ax.plot([], [], color="cyan", linewidth=1.5)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Motion (um)")
ax.grid(True, alpha=0.4)

# reasonable initial limits; y will autoscale if enabled
ax.set_xlim(0, PLOT_WINDOW_SEC if PLOT_WINDOW_SEC else 2.0)
ax.set_ylim(-50000, 50000)

time_counter = 0
redraw_counter = 0

def update(_frame):
    global time_counter, zi, redraw_counter

    if stop_event.is_set() and frame_q.empty():
        return (line,)

    # LOW_LATENCY: consume all and keep only latest frame
    raw = None
    while True:
        try:
            raw = frame_q.get_nowait()
        except queue.Empty:
            break

    if raw is None:
        return (line,)

    # Process one newest frame
    samples_v = bytes_to_volts(raw)
    ph = phase_deg_single_bin(samples_v)

    motion_um = ((ph + 180.0) / degrees_per_mm) * 1000.0

    time_counter += 1
    t = time_counter * ST

    y_filt, zi_local = lfilter(b, a, [motion_um], zi=zi)
    zi[:] = zi_local
    y_filt = float(y_filt[0])

    x_buf.append(t)
    y_buf.append(y_filt)

    # Update line
    x_arr = np.fromiter(x_buf, dtype=np.float32)
    y_arr = np.fromiter(y_buf, dtype=np.float32)
    line.set_data(x_arr, y_arr)

    # Sliding x-axis
    if PLOT_WINDOW_SEC is None:
        ax.set_xlim(0, max(1.0, float(x_arr[-1])))
    else:
        right = float(x_arr[-1])
        left = max(0.0, right - PLOT_WINDOW_SEC)
        ax.set_xlim(left, left + PLOT_WINDOW_SEC)

    # Y autoscale (throttled)
    if AUTOSCALE_Y:
        redraw_counter += 1
        if len(y_arr) > 5 and (redraw_counter % AUTOSCALE_EVERY == 0):
            y0 = float(np.min(y_arr))
            y1 = float(np.max(y_arr))
            span = max(y1 - y0, Y_MIN_SPAN)
            y0 -= Y_MARGIN * span
            y1 += Y_MARGIN * span
            ax.set_ylim(y0, y1)

    return (line,)

def on_close(_evt):
    stop_event.set()

fig.canvas.mpl_connect("close_event", on_close)

ani = FuncAnimation(
    fig,
    update,
    interval=int(1000 / TARGET_UI_FPS),
    blit=USE_BLIT
)

# ============================================================
# MAIN
# ============================================================
def main():
    print("OS:", platform.system())
    print(f"Fs={SAMPLING_FREQUENCY} Hz, N={NUMBER_OF_SAMPLES}, ST={ST:.6f}s, SweepFreq={SWEEP_FREQ:.2f}Hz")
    print(f"hz_per_m={hz_per_m}, PHASE_FREQ={PHASE_FREQ:.3f}Hz, f_step={f_step:.3f}Hz, target_bin={target_bin}")
    print(f"UI={TARGET_UI_FPS} FPS, window={PLOT_WINDOW_SEC}s, latency_mode={LOW_LATENCY_MODE}")

    if TEST_DEVICE != 1:
        raise RuntimeError("This combined version is written for TEST_DEVICE==1 (STM32H7 serial).")

    if platform.system() == "Linux":
        port = "/dev/ttyACM0"
    else:
        port = "/dev/tty.usbmodem3878386530331"

    ser = serial_init_specific(port)
    tx_dbm, tx_dbm_v, ver_major, ver_minor = configuration_process(ser)

    record_string, record_time_now = make_record_filename(ver_major, ver_minor)
    out_f, out_path = open_record_file(record_string)

    print("Recording to:", out_path)
    write_header(out_f, record_time_now, tx_dbm, tx_dbm_v)

    th = threading.Thread(target=acquisition_loop, args=(ser, out_f), daemon=True)
    th.start()

    plt.show()  # blocks until window closes

    # Cleanup
    stop_event.set()
    th.join(timeout=2.0)

    try:
        out_f.flush()
        out_f.close()
    except Exception:
        pass

    try:
        ser.close()
    except Exception:
        pass

    print("Exit.")

if __name__ == "__main__":
    main()
