import time
import pylibftdi as ftdi

DEVICE_ID = "FTBJ7TCT"
TEST_PACKET = b"1"

devices = ftdi.Driver().list_devices()
print("Detected:")
for d in devices:
    print(d)

dev = ftdi.Device(
    device_id=DEVICE_ID,
    mode="b",
    interface_select=ftdi.INTERFACE_A
)

# Put Channel A into FT245 synchronous FIFO mode
dev.ftdi_fn.ftdi_set_bitmode(0xFF, 0x40)
time.sleep(0.1)

dev.ftdi_fn.ftdi_read_data_set_chunksize(1024)
dev.ftdi_fn.ftdi_write_data_set_chunksize(1024)

dev.flush()

print("Packet length:", len(TEST_PACKET))

for i in range(64):
    dev.write(TEST_PACKET)
    print("sent", i)
    time.sleep(0.001)

#time.sleep(1.0)
dev.close()