import time
import pylibftdi as ftdi
import numpy as np

TEST_CONFIG_ONLY = 0
TEST_RADAR       = 1

DEVICE_ID = "FTBJ7TCT"

SYNCFF = 0x40
SIO_RTS_CTS_HS = (0x1 << 8)

READ_CHUNK_SIZE  = 0x10000
WRITE_CHUNK_SIZE = 0x10000

RX_IDLE_TIMEOUT = 2.0  # stop after 2 seconds with no RX data after RX starts

tx_start_char = b"C"

SWEEP_TIME = 1000e-6
SWEEP_GAP  = 100e-6

RECORD_TIME = 5

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
FIR_ENABLE       = 0
SEND_DATA_TYPE   = 1
ADC_RESOLUTION   = 16
SAMPLE_AVERAGING = 1


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

    push_u8("TX_MODE", TX_MODE)
    push_u8("GAIN", GAIN)
    push_u8("SWEEP_TYPE", SWEEP_TYPE)
    push_u8("DATA_LOG", DATA_LOG)
    push_u8("ADC_SELECT", ADC_SELECT)
    push_u8("USE_PLL", USE_PLL)
    push_u8("CHECK_MODE", FIR_ENABLE)
    push_u8("USB_DATA_TYPE", SEND_DATA_TYPE)
    push_u8("ADC_RESOLUTION", ADC_RESOLUTION)
    push_u8("SAMPLE_AVERAGING", SAMPLE_AVERAGING)

    return bytes(packet)


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

                if elapsed > 0:
                    throughput = (f.tell() / elapsed) / (1024 * 1024)
                else:
                    throughput = 0.0

                print(
                    f"RX {len(rx):6d} bytes   "
                    f"TOTAL {f.tell():10d} bytes   "
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

    f.close()
    dev.close()

    print("\nSAVED: record.bin")
    print("TOTAL BYTES:", total_bytes)