import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================
# SELECT MODE
# ============================================================

PLOT_ADC       = 1
PLOT_TEST_DATA = 0

# ============================================================
# USER SETTINGS
# ============================================================

FILENAME = "record.bin"

SAMPLES_PER_CHIRP = 500

START_CHIRP = 0
END_CHIRP   = None

CHIRP_STEP  = 10

FRAME_DELAY = 0.0001

# ============================================================
# READ RAW USB DATA
# ============================================================

with open(FILENAME, "rb") as f:
    raw = f.read()

valid_len = (len(raw) // 4) * 4
raw = raw[:valid_len]

# ============================================================
# TEST DATA MODE
# ============================================================

if PLOT_TEST_DATA == 1:

    words32 = np.frombuffer(raw, dtype=">u4")

    num_chirps = len(words32) // SAMPLES_PER_CHIRP

    usable_words = num_chirps * SAMPLES_PER_CHIRP

    words32 = words32[:usable_words]

    chirps = words32.reshape(
        num_chirps,
        SAMPLES_PER_CHIRP
    )

    print("TOTAL WORDS :", len(words32))
    print("NUM CHIRPS  :", num_chirps)

    if END_CHIRP is None:
        END_CHIRP = num_chirps

    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(SAMPLES_PER_CHIRP)

    line, = ax.plot(x, chirps[0])

    ax.set_xlabel("Sample in chirp")
    ax.set_ylabel("32-bit counter value")

    ax.grid(True)

    ax.set_ylim(
        0,
        np.max(chirps) + 10
    )

    for chirp_idx in range(
        START_CHIRP,
        END_CHIRP,
        CHIRP_STEP
    ):

        line.set_ydata(chirps[chirp_idx])

        ax.set_title(
            f"TEST DATA - Chirp "
            f"{chirp_idx + 1}/{num_chirps}"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        plt.pause(FRAME_DELAY)

    plt.ioff()
    plt.show()

# ============================================================
# ADC MODE
# ============================================================

if PLOT_ADC == 1:

    samples_u16 = np.frombuffer(raw, dtype=">u2")

    adc_a_u12 = samples_u16[0::2] & 0x0FFF
    adc_b_u12 = samples_u16[1::2] & 0x0FFF

    adc_a = adc_a_u12.astype(np.int32) - 2048
    adc_b = adc_b_u12.astype(np.int32) - 2048

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

    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(SAMPLES_PER_CHIRP)

    line_a, = ax.plot(x, adc_a_chirps[0], label="ADC A")
    line_b, = ax.plot(x, adc_b_chirps[0], label="ADC B")

    ax.set_xlabel("Sample in chirp")
    ax.set_ylabel("ADC value, centered 12-bit")

    ax.grid(True)
    ax.legend()

    y_min = min(adc_a.min(), adc_b.min())
    y_max = max(adc_a.max(), adc_b.max())

    ax.set_ylim(y_min - 100, y_max + 100)

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