# Select rx only process in top module of XC7A35T_FT2232H_Sycn
TEST_RX_ONLY            = 0

# Select rx tx process for these tests in top module of XC7A35T_FT2232H_Sycn
TEST_ECHO               = 0 
TEST_BENCHMARK          = 0 # RX + TX is 6.8 MB/sec so it is 6.8 x 2 = 13.6 MB/sec
TEST_BENCHMARK_CHECK    = 0 # RX + TX is 6.8 MB/sec so it is 6.8 x 2 = 13.6 MB/sec

# Select tx only process in top module of XC7A35T_FT2232H_Sycn
TEST_TX_ONLY            = 1 # 14.86 MB/sec

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

    dev.ftdi_fn.ftdi_set_latency_timer(1) # this is really important!!!!
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


if TEST_ECHO == 1:

    TEST_PACKET = b"0123456789" * 1000
    EXPECTED_LEN = len(TEST_PACKET)

    dev = open_ftdi()

    stale = drain_rx(dev)
    if stale:
        print("Drained stale RX:", stale)

    print("TX:", TEST_PACKET)
    print("TX length:", EXPECTED_LEN)

    dev.write(TEST_PACKET)

    rx = b""
    timeout_s = 1.0
    t0 = time.time()

    while len(rx) < EXPECTED_LEN and (time.time() - t0) < timeout_s:
        chunk = dev.read(EXPECTED_LEN - len(rx))
        if chunk:
            rx += chunk
        else:
            time.sleep(0.001)

    dev.close()

    print("RX:", rx)
    print("RX length:", len(rx))

    if rx == TEST_PACKET:
        print("ECHO OK")
    else:
        print("ECHO FAIL")
        print("Expected:", TEST_PACKET)
        print("Received:", rx)

if TEST_BENCHMARK_CHECK == 1:

    PACKET_SIZE = 8192 # 8192 fifo size and PACKET_SIZE 8192 works
    NUM_PACKETS = 1000
    TIMEOUT_S   = 10.0

    test_packet = bytes([i & 0xFF for i in range(PACKET_SIZE)])

    dev = open_ftdi()
    drain_rx(dev)

    total_rx = 0
    error = None

    t_start = time.perf_counter()

    for packet_idx in range(NUM_PACKETS):

        dev.write(test_packet)

        rx = b""
        t0 = time.perf_counter()

        while len(rx) < PACKET_SIZE:

            chunk = dev.read(PACKET_SIZE - len(rx))

            if chunk:
                rx += chunk
            else:
                if time.perf_counter() - t0 > TIMEOUT_S:
                    error = f"TIMEOUT packet={packet_idx}, received={len(rx)}"
                    break

                time.sleep(0.0001)

        if error:
            break

        if len(rx) != PACKET_SIZE:
            error = f"LENGTH packet={packet_idx}, received={len(rx)}"
            break

        if rx != test_packet:
            error = f"CORRUPT packet={packet_idx}"
            break

        total_rx += len(rx)

    elapsed = time.perf_counter() - t_start
    dev.close()

    print("RX bytes:", total_rx)
    print("Elapsed:", elapsed, "s")
    print("Throughput:", total_rx / elapsed / 1_000_000, "MB/s")

    if error is None:
        print("RESULT: BYTE RECEIVE OK")
    else:
        print("RESULT:", error)

if TEST_TX_ONLY == 1:

    START_COMMAND = b"1"

    READ_SIZE = 4096
    NUM_READS = 1000
    TIMEOUT_S = 5.0

    dev = open_ftdi()
    drain_rx(dev)

    dev.write(START_COMMAND)

    last_byte = None
    total_rx = 0
    error = None

    t_start = time.perf_counter()

    for read_idx in range(NUM_READS):

        rx = b""
        t0 = time.perf_counter()

        while len(rx) < READ_SIZE:

            chunk = dev.read(READ_SIZE - len(rx))

            if chunk:
                rx += chunk
            else:
                if time.perf_counter() - t0 > TIMEOUT_S:
                    error = f"TIMEOUT read={read_idx}, received={len(rx)}"
                    break

                time.sleep(0.0001)

        if error:
            break

        if len(rx) != READ_SIZE:
            error = f"LENGTH read={read_idx}, received={len(rx)}"
            break

        for b in rx:

            if last_byte is not None:
                expected = (last_byte + 1) & 0xFF

                if b != expected:
                    error = (
                        f"CORRUPT read={read_idx}, "
                        f"expected=0x{expected:02X}, got=0x{b:02X}"
                    )
                    break

            last_byte = b

        if error:
            break

        total_rx += len(rx)

    elapsed = time.perf_counter() - t_start
    dev.close()

    print("RX bytes:", total_rx)
    print("Elapsed:", elapsed, "s")

    if elapsed > 0:
        print("Throughput:", total_rx / elapsed / 1_000_000, "MB/s")

    if error is None:
        print("RESULT: TX ONLY OK")
    else:
        print("RESULT:", error)