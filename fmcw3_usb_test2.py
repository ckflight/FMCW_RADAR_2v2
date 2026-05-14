import time
import pylibftdi as ftdi
import numpy as np

TEST_CONFIG_ONLY = 0
TEST_RADAR       = 1

# =================================================
# COMMON FTDI DEFINITIONS
# =================================================

DEVICE_ID = "FTBJ7TCT"

SYNCFF = 0x40
SIO_RTS_CTS_HS = (0x1 << 8)

READ_CHUNK_SIZE  = 0x10000
WRITE_CHUNK_SIZE = 0x10000

# =================================================
# CONFIG VALUES
# =================================================

SWEEP_TIME = 1000e-6
SWEEP_GAP  = 1000e-6

RECORD_TIME = 10

SAMPLING_FREQUENCY = 2_000_000
NUMBER_OF_SAMPLES  = int(SAMPLING_FREQUENCY * SWEEP_TIME)

SWEEP_START = 5.20e9
SWEEP_BW    = 1000e6

TX_MODE          = 1
GAIN             = 10
SWEEP_TYPE       = 0
DATA_LOG         = 0
ADC_SELECT       = 0
USE_PLL          = 1
CHECK_MODE       = 4
USB_DATA_TYPE    = 1
ADC_RESOLUTION   = 16
SAMPLE_AVERAGING = 1

# =================================================
# FTDI INIT
# =================================================

def open_ftdi():

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

    print("FTDI OPENED")

    return dev

# =================================================
# CONFIG PACKET
# =================================================

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

    push_u8("TX_MODE", TX_MODE)
    push_u8("GAIN", GAIN)
    push_u8("SWEEP_TYPE", SWEEP_TYPE)
    push_u8("DATA_LOG", DATA_LOG)
    push_u8("ADC_SELECT", ADC_SELECT)
    push_u8("USE_PLL", USE_PLL)
    push_u8("CHECK_MODE", CHECK_MODE)
    push_u8("USB_DATA_TYPE", USB_DATA_TYPE)
    push_u8("ADC_RESOLUTION", ADC_RESOLUTION)
    push_u8("SAMPLE_AVERAGING", SAMPLE_AVERAGING)

    return bytes(packet)

# =================================================
# TEST CONFIG ONLY
# =================================================

if TEST_CONFIG_ONLY == 1:

    dev = open_ftdi()

    packet = build_packet()

    print("\nTOTAL BYTES:", len(packet))
    print("RAW HEX:", packet.hex(" "))

    print("\nSENDING CONFIG...\n")

    dev.write(packet)

    time.sleep(1.0)

    dev.close()

    print("DONE")

# =================================================
# TEST RADAR
# =================================================

if TEST_RADAR == 1:

    dev = open_ftdi()

    packet = build_packet()

    print("\nTOTAL CONFIG BYTES:", len(packet))
    print("RAW HEX:", packet.hex(" "))

    print("\nSENDING CONFIG...\n")

    dev.write(packet)

    time.sleep(1.0)

    print("SENDING C COMMAND...\n")

    dev.write(b"C")

    print("RECEIVING DATA...\n")

    f = open("record.bin", "wb")

    start_time = time.time()

    try:

        while True:

            rx = dev.read(512)

            if len(rx) > 0:

                f.write(rx)

                elapsed = time.time() - start_time

                throughput = (f.tell() / elapsed) / (1024 * 1024)

                print(
                    f"RX {len(rx):6d} bytes   "
                    f"TOTAL {f.tell():10d} bytes   "
                    f"{throughput:6.2f} MB/s"
                )

    except KeyboardInterrupt:

        print("\nSTOPPED")

    f.close()

    dev.close()

    print("\nSAVED: record.bin")