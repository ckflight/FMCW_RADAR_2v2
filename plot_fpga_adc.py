import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# READ RAW USB DATA
# ============================================================

with open("tx_stream.bin", "rb") as f:
    raw = f.read()

print("TOTAL BYTES:", len(raw))

# ============================================================
# CONVERT USB BYTES -> 16-bit ADC SAMPLES
# FPGA sends:
# adc_data_a & adc_data_b
#
# Byte order:
# [31:24] = A[15:8]
# [23:16] = A[7:0]
# [15:8 ] = B[15:8]
# [7 :0 ] = B[7:0]
# ============================================================

samples_u16 = np.frombuffer(raw, dtype=">u2")

# split channels
adc_a = samples_u16[0::2]
adc_b = samples_u16[1::2]

print("ADC A SAMPLES:", len(adc_a))
print("ADC B SAMPLES:", len(adc_b))

# ============================================================
# OPTIONAL: convert unsigned ADC to signed centered waveform
# if ADC midpoint is 32768
# ============================================================

adc_a = adc_a.astype(np.int32) - 32768
adc_b = adc_b.astype(np.int32) - 32768

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(adc_a[:2000], label="ADC A")
plt.plot(adc_b[:2000], label="ADC B")

plt.xlabel("Sample")
plt.ylabel("ADC Value")
plt.title("ADC Waveform")

plt.grid(True)
plt.legend()

plt.show()