import time
import pylibftdi as ftdi

DEVICE_ID = "FTBJ7TCT"
TEST_PACKET = b"==abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"

devices = ftdi.Driver().list_devices()
print("Detected:")
for d in devices:
    print(d)

dev = ftdi.Device(
    device_id=DEVICE_ID,
    mode="b",
    interface_select=ftdi.INTERFACE_A
)

dev.ftdi_fn.ftdi_read_data_set_chunksize(4096)
dev.ftdi_fn.ftdi_write_data_set_chunksize(4096)

dev.flush()

print("Packet length:", len(TEST_PACKET))

for i in range(5):
    dev.write(TEST_PACKET)
    print("sent", i)
    time.sleep(0.1)