import time
import pylibftdi as ftdi

DEVICE_ID = "FTBJ7TCT"

SYNCFF = 0x40
SIO_RTS_CTS_HS = (0x1 << 8)

TEST_PACKET1 = "1234512345123451234512345" # this one works

dev = ftdi.Device(
    device_id=DEVICE_ID,
    mode="b",
    interface_select=ftdi.INTERFACE_A
)

dev.open()
dev.ftdi_fn.ftdi_set_bitmode(0xFF, SYNCFF)
dev.ftdi_fn.ftdi_read_data_set_chunksize(0x10000)
dev.ftdi_fn.ftdi_write_data_set_chunksize(0x10000)
dev.ftdi_fn.ftdi_setflowctrl(SIO_RTS_CTS_HS)
dev.flush()

print("Packet1:", TEST_PACKET1)

dev.write(TEST_PACKET1)

time.sleep(1.0)
dev.close()