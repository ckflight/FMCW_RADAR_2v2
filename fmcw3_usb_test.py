import pylibftdi as ftdi

# -------------------------------
# FTDI INIT
# -------------------------------
def init_device():
    devices = ftdi.Driver().list_devices()
    for d in devices:
        print(d)

    dev = ftdi.Device(
        device_id="FTBJ7TCT",   # your FTDI serial
        mode='b',
        interface_select=ftdi.INTERFACE_A
    )
    dev.open()

    dev.ftdi_fn.ftdi_read_data_set_chunksize(4096)
    dev.ftdi_fn.ftdi_write_data_set_chunksize(4096)

    dev.flush()
    return dev


# -------------------------------
# TEST PACKET (42 BYTES)
# -------------------------------
def send_test(dev):
    data = b"==abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"

    print("Sending:", data)
    print("Length :", len(data))  # MUST be 42

    dev.write(data)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    dev = init_device()
    send_test(dev)