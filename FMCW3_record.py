import time
import struct
import numpy as np

# =========================================================
# OS SELECT
# =========================================================
OS_TYPE = "windows"   # "ubuntu" or "windows"

if OS_TYPE == "ubuntu":
    import pylibftdi as ftdi
elif OS_TYPE == "windows":
    import ftd2xx as ftdi
else:
    raise ValueError("OS_TYPE must be 'ubuntu' or 'windows'")

TEST_MUX = 0

if TEST_MUX == 0:
    SWEEP_TIME = 500e-6
    SWEEP_GAP  = 10e-6
else:
    SWEEP_TIME = 250e-6
    SWEEP_GAP  = 10e-6

USE_LOG_NAME = False
CURRENT_FIR_HPF_KHZ = "400KHz"
OUTPUT_FILENAME = CURRENT_FIR_HPF_KHZ + "_400mhz.bin"

RECORD_TIME = 5

SAMPLING_FREQUENCY = 2_000_000
NUMBER_OF_SAMPLES  = int(SAMPLING_FREQUENCY * SWEEP_TIME)

SWEEP_START = 5.30e9
SWEEP_BW    = 700e6

GAIN             = 10
SWEEP_TYPE       = 0
DATA_LOG         = 0
ADC_SELECT       = 0
PA_MODE          = 0
FIR_ENABLE       = 1
SEND_DATA_TYPE   = 1
ADC_RESOLUTION   = 12
SAMPLE_AVERAGING = 1

INFO_SECTOR_SIZE = 512

RX_IDLE_TIMEOUT = 2.0
TX_START_CHAR = b"C"

def open_ftdi():

    DEVICE_ID = "FTBJ7TCT"
    DEVICE_INDEX = 0

    SYNCFF = 0x40
    SIO_RTS_CTS_HS = (0x1 << 8)

    READ_CHUNK_SIZE  = 0x10000
    WRITE_CHUNK_SIZE = 0x10000

    if OS_TYPE == "ubuntu":

        dev = ftdi.Device(
            device_id=DEVICE_ID,
            mode="b",
            interface_select=ftdi.INTERFACE_A
        )

        dev.open()

        dev.ftdi_fn.ftdi_set_bitmode(0xFF, SYNCFF)

        time.sleep(0.1)

        dev.ftdi_fn.ftdi_read_data_set_chunksize(READ_CHUNK_SIZE)
        dev.ftdi_fn.ftdi_write_data_set_chunksize(WRITE_CHUNK_SIZE)

        dev.ftdi_fn.ftdi_setflowctrl(SIO_RTS_CTS_HS)

        dev.flush()

    elif OS_TYPE == "windows":

        dev = ftdi.open(DEVICE_INDEX)

        dev.setBitMode(0xFF, SYNCFF)

        time.sleep(0.1)

        dev.setUSBParameters(
            READ_CHUNK_SIZE,
            WRITE_CHUNK_SIZE
        )

        dev.setLatencyTimer(2)

        dev.setFlowControl(
            ftdi.defines.FLOW_RTS_CTS,
            0,
            0
        )

        dev.purge()

    print(f"FTDI OPENED using {OS_TYPE}")

    return dev


def build_packet():

    packet = bytearray()

    def push_u8(name, value):
        value = int(value) & 0xFF
        print(f"{name:<18} 0x{value:02X}")
        packet.append(value)

    def push_u16(name, value):
        value = int(value) & 0xFFFF

        msb = (value >> 8) & 0xFF
        lsb = value & 0xFF

        print(f"{name:<18} 0x{msb:02X} 0x{lsb:02X} ({value})")

        packet.append(msb)
        packet.append(lsb)

    print("\nTX CONFIG PACKET:")
    print("HEADER             0x3D 0x3D")

    packet += b"=="

    push_u16("SWEEP_TIME", np.uint16(SWEEP_TIME * 1e6))
    push_u16("SWEEP_GAP", np.uint16(SWEEP_GAP * 1e6))

    push_u8("RECORD_TIME", np.uint8(RECORD_TIME))

    push_u16("FS_KHZ", np.uint16(SAMPLING_FREQUENCY / 1e3))
    push_u16("NUM_SAMPLES", np.uint16(NUMBER_OF_SAMPLES))

    push_u16("SWEEP_START", np.uint16(SWEEP_START / 1e7))
    push_u16("SWEEP_BW", np.uint16(SWEEP_BW / 1e6))

    push_u8("TEST_MUX", TEST_MUX)
    push_u8("GAIN", GAIN)
    push_u8("SWEEP_TYPE", SWEEP_TYPE)
    push_u8("DATA_LOG", DATA_LOG)
    push_u8("ADC_SELECT", ADC_SELECT)
    push_u8("USE_PLL", PA_MODE)
    push_u8("CHECK_MODE", FIR_ENABLE)
    push_u8("USB_DATA_TYPE", SEND_DATA_TYPE)
    push_u8("ADC_RESOLUTION", ADC_RESOLUTION)
    push_u8("SAMPLE_AVERAGING", SAMPLE_AVERAGING)

    return bytes(packet)


def build_info_sector():

    info = bytearray(INFO_SECTOR_SIZE)

    offset = 0

    def put_u32(value):
        nonlocal offset
        struct.pack_into("<I", info, offset, int(value))
        offset += 4

    def put_f32(value):
        nonlocal offset
        struct.pack_into("<f", info, offset, float(value))
        offset += 4

    hz_per_m = (2.0 * SWEEP_BW) / (3.0e8 * SWEEP_TIME)

    info[0:4] = b"FMCW"
    offset = 4

    put_u32(1)

    put_f32(SWEEP_TIME)
    put_f32(SWEEP_GAP)
    put_u32(RECORD_TIME)

    put_u32(SAMPLING_FREQUENCY)
    put_u32(NUMBER_OF_SAMPLES)

    put_f32(SWEEP_START)
    put_f32(SWEEP_BW)

    put_u32(TEST_MUX)
    put_u32(GAIN)
    put_u32(SWEEP_TYPE)
    put_u32(DATA_LOG)
    put_u32(ADC_SELECT)
    put_u32(PA_MODE)
    put_u32(FIR_ENABLE)
    put_u32(SEND_DATA_TYPE)
    put_u32(ADC_RESOLUTION)
    put_u32(SAMPLE_AVERAGING)

    put_f32(hz_per_m)

    put_u32(INFO_SECTOR_SIZE)
    put_u32(INFO_SECTOR_SIZE)

    return bytes(info)


dev = open_ftdi()

packet = build_packet()

print("\nTOTAL CONFIG BYTES:", len(packet))
print("RAW HEX:", packet.hex(" "))

print("\nSENDING CONFIG...\n")
dev.write(packet)

time.sleep(1.0)

print("SENDING C COMMAND...\n")
dev.write(TX_START_CHAR)

print("RECEIVING DATA...\n")

if USE_LOG_NAME == True:
    f = open(OUTPUT_FILENAME, "wb")
    saved_name = OUTPUT_FILENAME
else:
    f = open("record.bin", "wb")
    saved_name = "record.bin"

f.write(build_info_sector())
f.flush()

start_time = time.time()
last_rx_time = None
received_started = False

try:

    while True:

        now = time.time()

        if now - start_time >= RECORD_TIME:
            print(f"\nRECORD_TIME reached: {RECORD_TIME} s. Ending reception.")
            break

        rx = dev.read(1024)
        now = time.time()

        if len(rx) > 0:

            received_started = True
            last_rx_time = now

            f.write(rx)

            elapsed = now - start_time
            data_bytes = f.tell() - INFO_SECTOR_SIZE

            if elapsed > 0:
                throughput = (data_bytes / elapsed) / (1024 * 1024)
            else:
                throughput = 0.0

            print(
                f"RX {len(rx):6d} bytes   "
                f"DATA {data_bytes:10d} bytes   "
                f"{throughput:6.2f} MB/s"
            )

        else:

            if received_started and last_rx_time is not None:
                if now - last_rx_time >= RX_IDLE_TIMEOUT:
                    print(
                        f"\nRX idle for {RX_IDLE_TIMEOUT:.1f} s. "
                        "Ending reception."
                    )
                    break

            time.sleep(0.001)

except KeyboardInterrupt:

    print("\nSTOPPED BY USER")

total_bytes = f.tell()
data_bytes = total_bytes - INFO_SECTOR_SIZE

f.close()
dev.close()

print(f"\nSAVED: {saved_name}")
print("TOTAL BYTES:", total_bytes)
print("DATA BYTES :", data_bytes)