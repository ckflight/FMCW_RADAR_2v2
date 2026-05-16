import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# USER SETTINGS
# ============================================================

FILENAME = "record.bin"

SAMPLES_PER_CHIRP = 500

START_CHIRP = 0
END_CHIRP   = None

CHIRP_STEP  = 10        # plot every Nth chirp

FRAME_DELAY = 0.0001

# ============================================================
# READ RAW USB DATA
# ============================================================

with open(FILENAME, "rb") as f:
    raw = f.read()

valid_len = (len(raw) // 4) * 4
raw = raw[:valid_len]

samples_u16 = np.frombuffer(raw, dtype=">u2")

# FPGA sends:
# A[15:0], B[15:0]

adc_a_u12 = samples_u16[0::2] & 0x0FFF
adc_b_u12 = samples_u16[1::2] & 0x0FFF

# convert 12-bit offset binary to centered signed
adc_a = adc_a_u12.astype(np.int32) - 2048
adc_b = adc_b_u12.astype(np.int32) - 2048

# ============================================================
# RESHAPE INTO CHIRPS
# ============================================================

num_chirps = len(adc_a) // SAMPLES_PER_CHIRP

usable_samples = num_chirps * SAMPLES_PER_CHIRP

adc_a_chirps = adc_a[:usable_samples].reshape(
    num_chirps,
    SAMPLES_PER_CHIRP
)

adc_b_chirps = adc_b[:usable_samples].reshape(
    num_chirps,
    SAMPLES_PER_CHIRP
)

print("TOTAL BYTES :", len(raw))
print("NUM CHIRPS  :", num_chirps)

if END_CHIRP is None:
    END_CHIRP = num_chirps

# ============================================================
# VIDEO-LIKE PLOT
# ============================================================

plt.ion()

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(SAMPLES_PER_CHIRP)

line_a, = ax.plot(x, adc_a_chirps[0], label="ADC A")
line_b, = ax.plot(x, adc_b_chirps[0], label="ADC B")

ax.set_xlabel("Sample in chirp")
ax.set_ylabel("ADC value, centered 12-bit")
ax.grid(True)
ax.legend()

# fixed axis for stable animation
y_min = min(adc_a.min(), adc_b.min())
y_max = max(adc_a.max(), adc_b.max())

ax.set_ylim(y_min - 100, y_max + 100)

# ============================================================
# PLAY CHIRPS
# ============================================================

for chirp_idx in range(
    START_CHIRP,
    END_CHIRP,
    CHIRP_STEP
):

    line_a.set_ydata(adc_a_chirps[chirp_idx])
    line_b.set_ydata(adc_b_chirps[chirp_idx])

    ax.set_title(
        f"ADC Waveform - Chirp "
        f"{chirp_idx + 1}/{num_chirps}"
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(FRAME_DELAY)

plt.ioff()
plt.show()