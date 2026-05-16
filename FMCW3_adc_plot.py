import numpy as np
import matplotlib.pyplot as plt

PLOT_ADC       = 1
PLOT_TEST_DATA = 0

FILENAME = "record.bin"
HEADER = b"\xC8\xC8\xC8\xC8"

SAMPLES_PER_CHIRP = 1000
START_CHIRP = 0
END_CHIRP   = None
CHIRP_STEP  = 1
FRAME_DELAY = 0.001

ADC_BITS = 12
ADC_FULL_SCALE_VOLTAGE = 3.3

BYTES_PER_SAMPLE_PAIR = 4
BYTES_PER_CHIRP = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE_PAIR

with open(FILENAME, "rb") as f:
    raw_all = f.read()

header_positions = []
idx = 0

while True:
    idx = raw_all.find(HEADER, idx)
    if idx < 0:
        break
    header_positions.append(idx)
    idx += len(HEADER)

print("RAW BYTES     :", len(raw_all))
print("HEADERS FOUND :", len(header_positions))

if len(header_positions) == 0:
    raise RuntimeError("No header found")

chirp_payloads = []

for h_idx in header_positions:
    start = h_idx + len(HEADER)
    end   = start + BYTES_PER_CHIRP

    if end <= len(raw_all):
        chirp_payloads.append(raw_all[start:end])

print("VALID CHIRPS  :", len(chirp_payloads))

if len(chirp_payloads) == 0:
    raise RuntimeError("No complete chirp found after headers")

if PLOT_TEST_DATA == 1:

    chirps = []

    for chirp in chirp_payloads:
        words32 = np.frombuffer(chirp, dtype=">u4")
        chirps.append(words32)

    chirps = np.array(chirps)
    num_chirps = chirps.shape[0]

    print("NUM CHIRPS  :", num_chirps)
    print("WORDS/CHIRP :", chirps.shape[1])

    if END_CHIRP is None:
        END_CHIRP = num_chirps

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(SAMPLES_PER_CHIRP)
    line, = ax.plot(x, chirps[0])

    ax.set_xlabel("Sample in chirp")
    ax.set_ylabel("32-bit counter value")
    ax.grid(True)
    ax.set_ylim(0, np.max(chirps) + 10)

    for chirp_idx in range(START_CHIRP, END_CHIRP, CHIRP_STEP):
        line.set_ydata(chirps[chirp_idx])
        ax.set_title(f"TEST DATA - Chirp {chirp_idx + 1}/{num_chirps}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(FRAME_DELAY)

    plt.ioff()
    plt.show()

if PLOT_ADC == 1:

    adc_a_chirps = []
    adc_b_chirps = []

    adc_max = (2 ** ADC_BITS) - 1

    for chirp in chirp_payloads:

        samples_u16 = np.frombuffer(chirp, dtype=">u2")

        adc_a_u12 = samples_u16[0::2] & 0x0FFF
        adc_b_u12 = samples_u16[1::2] & 0x0FFF

        adc_a_voltage = (
            adc_a_u12.astype(np.float32) / adc_max
        ) * ADC_FULL_SCALE_VOLTAGE

        adc_b_voltage = (
            adc_b_u12.astype(np.float32) / adc_max
        ) * ADC_FULL_SCALE_VOLTAGE

        adc_a_chirps.append(adc_a_voltage)
        adc_b_chirps.append(adc_b_voltage)

    adc_a_chirps = np.array(adc_a_chirps)
    adc_b_chirps = np.array(adc_b_chirps)

    num_chirps = adc_a_chirps.shape[0]

    print("NUM CHIRPS :", num_chirps)
    print("SAMPLES/CH :", adc_a_chirps.shape[1])

    if END_CHIRP is None:
        END_CHIRP = num_chirps

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(SAMPLES_PER_CHIRP)

    line_a, = ax.plot(x, adc_a_chirps[0], label="ADC A")
    line_b, = ax.plot(x, adc_b_chirps[0], label="ADC B")

    ax.set_xlabel("Sample in chirp")
    ax.set_ylabel("ADC Voltage (V)")
    ax.grid(True)
    ax.legend()

    y_min = min(adc_a_chirps.min(), adc_b_chirps.min())
    y_max = max(adc_a_chirps.max(), adc_b_chirps.max())

    ax.set_ylim(y_min - 0.05, y_max + 0.05)

    for chirp_idx in range(START_CHIRP, END_CHIRP, CHIRP_STEP):

        line_a.set_ydata(adc_a_chirps[chirp_idx])
        line_b.set_ydata(adc_b_chirps[chirp_idx])

        ax.set_title(f"ADC Voltage - Chirp {chirp_idx + 1}/{num_chirps}")

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(FRAME_DELAY)

    plt.ioff()
    plt.show()