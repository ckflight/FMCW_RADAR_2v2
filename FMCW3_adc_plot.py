import numpy as np
import matplotlib.pyplot as plt
import struct

PLOT_ADC       = 1
PLOT_TEST_DATA = 0

FILENAME = "record.bin"
HEADER = b"\xC8\xC8\xC8\xC8"

INFO_SECTOR_SIZE = 512

START_CHIRP = 0
END_CHIRP   = None
CHIRP_STEP  = 1
FRAME_DELAY = 0.001

ADC_BITS = 12
ADC_FULL_SCALE_VOLTAGE = 3.0

BYTES_PER_SAMPLE_PAIR = 4

def print_info(info):
    print("\nINFO SECTOR")
    print("-----------")

    readable = {
        "VERSION":              info["VERSION"],
        "SWEEP_TIME":           f"{info['SWEEP_TIME'] * 1e6:.0f} us",
        "SWEEP_GAP":            f"{info['SWEEP_GAP'] * 1e6:.0f} us",
        "RECORD_TIME":          f"{info['RECORD_TIME']} s",
        "SAMPLING_FREQUENCY":   f"{info['SAMPLING_FREQUENCY'] / 1e6:.3f} MHz",
        "NUMBER_OF_SAMPLES":    info["NUMBER_OF_SAMPLES"],
        "SWEEP_START":          f"{info['SWEEP_START'] / 1e6:.0f} MHz",
        "SWEEP_BW":             f"{info['SWEEP_BW'] / 1e6:.0f} MHz",
        "TX_MODE":              info["TX_MODE"],
        "GAIN":                 f"{info['GAIN']} dB",
        "SWEEP_TYPE":           info["SWEEP_TYPE"],
        "DATA_LOG":             info["DATA_LOG"],
        "ADC_SELECT":           info["ADC_SELECT"],
        "USE_PLL":              info["USE_PLL"],
        "FIR_ENABLE":           info["FIR_ENABLE"],
        "SEND_DATA_TYPE":       info["SEND_DATA_TYPE"],
        "ADC_RESOLUTION":       f"{info['ADC_RESOLUTION']} bit",
        "SAMPLE_AVERAGING":     info["SAMPLE_AVERAGING"],
        "HZ_PER_M":             f"{info['HZ_PER_M']:.1f} Hz/m",
        "INFO_SECTOR_SIZE":     f"{info['INFO_SECTOR_SIZE']} bytes",
        "DATA_START_BYTE":      info["DATA_START_BYTE"],
    }

    for k, v in readable.items():
        print(f"{k:<22}: {v}")

def parse_info_sector(info):

    if info[0:4] != b"FMCW":
        raise RuntimeError("Invalid info sector magic. Expected FMCW.")

    offset = 4

    def get_u32():
        nonlocal offset
        value = struct.unpack_from("<I", info, offset)[0]
        offset += 4
        return value

    def get_f32():
        nonlocal offset
        value = struct.unpack_from("<f", info, offset)[0]
        offset += 4
        return value

    parsed = {}

    parsed["VERSION"] = get_u32()

    parsed["SWEEP_TIME"] = get_f32()
    parsed["SWEEP_GAP"] = get_f32()
    parsed["RECORD_TIME"] = get_u32()

    parsed["SAMPLING_FREQUENCY"] = get_u32()
    parsed["NUMBER_OF_SAMPLES"] = get_u32()

    parsed["SWEEP_START"] = get_f32()
    parsed["SWEEP_BW"] = get_f32()

    parsed["TX_MODE"] = get_u32()
    parsed["GAIN"] = get_u32()
    parsed["SWEEP_TYPE"] = get_u32()
    parsed["DATA_LOG"] = get_u32()
    parsed["ADC_SELECT"] = get_u32()
    parsed["USE_PLL"] = get_u32()
    parsed["FIR_ENABLE"] = get_u32()
    parsed["SEND_DATA_TYPE"] = get_u32()
    parsed["ADC_RESOLUTION"] = get_u32()
    parsed["SAMPLE_AVERAGING"] = get_u32()

    parsed["HZ_PER_M"] = get_f32()

    parsed["INFO_SECTOR_SIZE"] = get_u32()
    parsed["DATA_START_BYTE"] = get_u32()

    return parsed


with open(FILENAME, "rb") as f:
    raw_all = f.read()

info_raw = raw_all[:INFO_SECTOR_SIZE]
raw_data = raw_all[INFO_SECTOR_SIZE:]

info = parse_info_sector(info_raw)

print_info(info)

SAMPLES_PER_CHIRP = int(info["NUMBER_OF_SAMPLES"])
ADC_BITS = int(info["ADC_RESOLUTION"])

BYTES_PER_CHIRP = SAMPLES_PER_CHIRP * BYTES_PER_SAMPLE_PAIR

print("\nDERIVED")
print("-------")
print("SAMPLES_PER_CHIRP   :", SAMPLES_PER_CHIRP)
print("BYTES_PER_CHIRP     :", BYTES_PER_CHIRP)
print("ADC_BITS            :", ADC_BITS)
print("RAW FILE BYTES      :", len(raw_all))
print("RAW DATA BYTES      :", len(raw_data))

header_positions = []
idx = 0

while True:
    idx = raw_data.find(HEADER, idx)
    if idx < 0:
        break

    header_positions.append(idx)
    idx += len(HEADER)

print("\nHEADERS FOUND       :", len(header_positions))

if len(header_positions) == 0:
    raise RuntimeError("No header found")

chirp_payloads = []

for h_idx in header_positions:
    start = h_idx + len(HEADER)
    end   = start + BYTES_PER_CHIRP

    if end <= len(raw_data):
        chirp_payloads.append(raw_data[start:end])

print("VALID CHIRPS        :", len(chirp_payloads))

if len(chirp_payloads) == 0:
    raise RuntimeError("No complete chirp found after headers")


if PLOT_TEST_DATA == 1:

    chirps = []

    for chirp in chirp_payloads:
        words32 = np.frombuffer(chirp, dtype=">u4")
        chirps.append(words32)

    chirps = np.array(chirps)
    num_chirps = chirps.shape[0]

    print("NUM CHIRPS          :", num_chirps)
    print("WORDS/CHIRP         :", chirps.shape[1])

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

    print("NUM CHIRPS          :", num_chirps)
    print("SAMPLES/CH          :", adc_a_chirps.shape[1])

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