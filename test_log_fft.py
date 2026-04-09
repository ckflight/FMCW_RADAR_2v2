import numpy as np
import matplotlib.pyplot as plt

OPERATING_SYSTEM = 2   # 1 = Ubuntu/Linux, 2 = Windows

if OPERATING_SYSTEM == 1:  # Ubuntu
    BIN_FILE = "/home/ck/Desktop/flight_log.bin"
elif OPERATING_SYSTEM == 2:  # Windows
    BIN_FILE = r"C:\Users\CK\Desktop\flight_log.bin"
    
FS = 3_720_000
SAMPLES_PER_CHIRP = 900
ADC_BITS = 16

# read file
data = np.fromfile(BIN_FILE, dtype='<u2')

# keep only full chirps
num_chirps = len(data) // SAMPLES_PER_CHIRP
data = data[:num_chirps * SAMPLES_PER_CHIRP]
chirps = data.reshape(num_chirps, SAMPLES_PER_CHIRP)

print("Total samples :", len(data))
print("Num chirps    :", num_chirps)
print("Samples/chirp :", SAMPLES_PER_CHIRP)

# frequency axis
f = np.fft.rfftfreq(SAMPLES_PER_CHIRP, d=1.0 / FS) / 1e6

# full-scale peak reference for signed 16-bit style normalization
FS_PEAK = 2**(ADC_BITS - 1)   # 32768

plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

# first chirp init
x = chirps[0].astype(np.float32)

# remove DC
x = x - np.mean(x)

# window
w = np.hanning(len(x))
xw = x * w

# coherent gain correction
cg = np.sum(w) / len(w)

# normalize to full scale
xw_fs = xw / FS_PEAK

# FFT
X = np.fft.rfft(xw_fs)

# dBFS per FFT bin
mag_dbfs = 20 * np.log10((np.abs(X) / (len(x) * cg / 2)) + 1e-20)

line, = ax.plot(f, mag_dbfs)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Magnitude (dBFS)")
ax.set_title("FFT of Chirp 0 (dBFS)")
ax.grid(True)
ax.set_ylim(-160, 0)

for i in range(num_chirps):
    x = chirps[i].astype(np.float32)

    # remove DC
    x = x - np.mean(x)

    # window
    xw = x * w

    # normalize to full scale
    xw_fs = xw / FS_PEAK

    # FFT
    X = np.fft.rfft(xw_fs)

    # dBFS per FFT bin
    mag_dbfs = 20 * np.log10((np.abs(X) / (len(x) * cg / 2)) + 1e-20)

    line.set_ydata(mag_dbfs)
    ax.set_title(f"FFT of Chirp {i} (dBFS)")

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.0002)

plt.ioff()
plt.show()