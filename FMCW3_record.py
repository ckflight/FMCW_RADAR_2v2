import time
import struct
import pylibftdi as ftdi
import numpy as np

TEST_CONFIG_ONLY = 0
TEST_RADAR       = 1

DEVICE_ID = "FTBJ7TCT"

SYNCFF = 0x40
SIO_RTS_CTS_HS = (0x1 << 8)

READ_CHUNK_SIZE  = 0x10000
WRITE_CHUNK_SIZE = 0x10000

RX_IDLE_TIMEOUT = 2.0

tx_start_char = b"C"
TEST_MUX         = 0 # 1 generate a test mux to test project, 0 adf4158 generated mux

# ADF4158 Setting
if TEST_MUX == 0:
    SWEEP_TIME = 250e-6
    SWEEP_GAP  = 10e-6

# Test mux settings
else:
    SWEEP_TIME = 250e-6 # fpga sets this. this number has no affect
    SWEEP_GAP  = 10e-6

RECORD_TIME = 5

SAMPLING_FREQUENCY = 2_000_000
NUMBER_OF_SAMPLES  = int(SAMPLING_FREQUENCY * SWEEP_TIME)

SWEEP_START = 5.20e9
SWEEP_BW    = 900e6

GAIN             = 10
SWEEP_TYPE       = 0
DATA_LOG         = 0
ADC_SELECT       = 0
PA_MODE          = 0 # 1 on off, 0 on during chirp
FIR_ENABLE       = 0
SEND_DATA_TYPE   = 0 # 1 adc, 0 test data
ADC_RESOLUTION   = 16
SAMPLE_AVERAGING = 1

INFO_SECTOR_SIZE = 512


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

    # 0-3: file magic
    info[0:4] = b"FMCW"
    offset = 4

    # 4-7: binary format version
    put_u32(1)

    # Radar config
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

    # File layout info
    put_u32(INFO_SECTOR_SIZE)
    put_u32(INFO_SECTOR_SIZE)

    return bytes(info)


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


if TEST_RADAR == 1:

    dev = open_ftdi()

    packet = build_packet()

    print("\nTOTAL CONFIG BYTES:", len(packet))
    print("RAW HEX:", packet.hex(" "))

    print("\nSENDING CONFIG...\n")

    dev.write(packet)

    time.sleep(1.0)

    print("SENDING C COMMAND...\n")

    dev.write(tx_start_char)

    print("RECEIVING DATA...\n")

    f = open("record.bin", "wb")

    # Write 512-byte binary info sector first
    f.write(build_info_sector())
    f.flush()

    start_time = time.time()
    last_rx_time = None
    received_started = False

    try:

        while True:

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

    print("\nSAVED: record.bin")
    print("TOTAL BYTES:", total_bytes)
    print("DATA BYTES :", data_bytes)