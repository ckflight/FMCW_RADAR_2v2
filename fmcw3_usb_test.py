# Select rx only process in top module of XC7A35T_FT2232H_Sycn
TEST_RX_ONLY            = 0

# Select rx tx process for these tests in top module of XC7A35T_FT2232H_Sycn
TEST_ECHO               = 0 
TEST_BENCHMARK          = 0 # RX + TX is 6.8 MB/sec so it is 6.8 x 2 = 13.6 MB/sec
TEST_BENCHMARK_CHECK    = 0 # RX + TX is 6.8 MB/sec so it is 6.8 x 2 = 13.6 MB/sec

# Select tx only process in top module of XC7A35T_FT2232H_Sycn
# IMPORTANT NOTE: TX ONLY WORKS. Just after sometime data sync shifts so each 4096 etc read does not start 0x00 but if you follow
# multiple read print it is seen that next 4096 reception starts with the incremented data!
TEST_TX_ONLY            = 0 # 14.86 MB/sec
TEST_TX_ONLY1           = 0 # prints received data simple to check 512 x 10 reception etc
TEST_TX_ONLY2           = 1 # writes data to bin file then checks byte by byte to see if it is incrementing correctly

import time
import pylibftdi as ftdi

DEVICE_ID = "FTBJ7TCT"

SYNCFF = 0x40
SIO_RTS_CTS_HS = (0x1 << 8)


def drain_rx(dev, duration_s=0.05):
    drained = b""
    t0 = time.time()

    while time.time() - t0 < duration_s:
        chunk = dev.read(4096)
        if chunk:
            drained += chunk
            t0 = time.time()
        else:
            time.sleep(0.001)

    return drained


def open_ftdi():
    dev = ftdi.Device(
        device_id=DEVICE_ID,
        mode="b",
        interface_select=ftdi.INTERFACE_A
    )

    dev.open()

    dev.ftdi_fn.ftdi_set_bitmode(0xFF, SYNCFF)
    time.sleep(0.1)

    dev.ftdi_fn.ftdi_set_latency_timer(10) # this is really important!!!!
    dev.ftdi_fn.ftdi_read_data_set_chunksize(0x10000)
    dev.ftdi_fn.ftdi_write_data_set_chunksize(0x10000)
    dev.ftdi_fn.ftdi_setflowctrl(SIO_RTS_CTS_HS)

    dev.flush()
    return dev


if TEST_RX_ONLY == 1:

    TEST_PACKET1 = b"123456789"

    dev = open_ftdi()

    print("Packet1:", TEST_PACKET1)

    dev.write(TEST_PACKET1)
    dev.write(TEST_PACKET1)
    dev.write(TEST_PACKET1)

    time.sleep(1.0)
    dev.close()

if TEST_TX_ONLY1 == 1:

    START_COMMAND = b"1"

    READ_SIZE = 1024
    NUM_READS = 100

    dev = open_ftdi()
    drain_rx(dev)

    # tell FPGA to start TX stream
    dev.write(START_COMMAND)

    print("Receiving TX stream...\n")

    for read_idx in range(NUM_READS):

        rx = dev.read(READ_SIZE)

        if rx:

            print(f"READ {read_idx}")
            print("RX LEN:", len(rx))
            print(rx.hex(" "))

        else:

            print(f"READ {read_idx} -> NO DATA")
            time.sleep(0.001)

    dev.close()

if TEST_TX_ONLY2 == 1:

    START_COMMAND = b"1"

    TOTAL_READ_SIZE = 1024 * 125
    READ_SIZE = 4096
    NUM_READS = int(TOTAL_READ_SIZE / READ_SIZE)
    NUM_OF_REPEAT  = 1000
    OUTPUT_FILE = "tx_stream.bin"

    dev = open_ftdi()
    drain_rx(dev)

    dev.flush()
    for rp in range(NUM_OF_REPEAT):
        
        # start FPGA TX stream
        dev.write(START_COMMAND)

        print("Receiving data...\n")

        t_start = time.perf_counter()

        with open(OUTPUT_FILE, "wb") as f:

            total_bytes = 0

            for read_idx in range(NUM_READS):

                rx = b""

                while len(rx) < READ_SIZE:

                    chunk = dev.read(READ_SIZE - len(rx))

                    if chunk:
                        rx += chunk
                    else:
                        time.sleep(0.0001)

                f.write(rx)
                total_bytes += len(rx)

        t_end = time.perf_counter()

        elapsed_s = t_end - t_start
        throughput_MBps = total_bytes / elapsed_s / (1024 * 1024)

        print(f"RX TIME     : {elapsed_s:.6f} s")
        print(f"THROUGHPUT  : {throughput_MBps:.2f} MB/s")

        
        print("\nSaved:", OUTPUT_FILE)

        # ---------------------------------------------------
        # VERIFY INCREMENT CONTINUITY
        # ---------------------------------------------------

        errors = 0

        with open(OUTPUT_FILE, "rb") as f:

            data = f.read()

            # ---------------------------------------------------
            # CHECK START / END BYTE
            # ---------------------------------------------------

            if len(data) > 0:

                first_byte = data[0]
                last_byte  = data[-1]

                print(f"FIRST BYTE : 0x{first_byte:02X}")
                print(f"LAST BYTE  : 0x{last_byte:02X}")

                if first_byte != 0x00:
                    print("ERROR: STREAM DOES NOT START WITH 0x00***********************************************************")
                    errors += 1

                if last_byte != 0xFF:
                    print("ERROR: STREAM DOES NOT END WITH 0xFF***********************************************************")
                    errors += 1

        print("Checking increment continuity...\n")


        previous_byte = None

        for i, current_byte in enumerate(data):

            if previous_byte is not None:

                expected = (previous_byte + 1) & 0xFF

                if current_byte != expected:

                    errors += 1

                    print("\nERROR")
                    print(f"BYTE INDEX : {i}")
                    print(f"PREVIOUS   : 0x{previous_byte:02X}")
                    print(f"EXPECTED   : 0x{expected:02X}")
                    print(f"GOT        : 0x{current_byte:02X}")

            previous_byte = current_byte



        if errors == 0:
            print("\n----- SUMMARY -----")
            print("TOTAL BYTES :", len(data))
            print("RESULT      : STREAM OK")
            print("ERRORS      :", errors)

            
        else:
            print("RESULT      : STREAM HAS ERRORS***********************************************************")

        print("REPEAT COUNTER: ", rp)

    dev.close()