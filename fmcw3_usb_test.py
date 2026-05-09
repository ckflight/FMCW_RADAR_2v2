TEST_RX_ONLY    = 0
TEST_ECHO       = 1
TEST_BENCHMARK  = 0 # RX + TX is 6.8 MB/sec so it is 6.8 x 2 = 13.6 MB/sec
TEST_TX_ONLY    = 0 # 14.86 MB/sec

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

    TEST_PACKET = b"123456789ABCDEFGHJKL" * 200
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


if TEST_BENCHMARK == 1:

    PACKET_SIZE = 6400 # rx tx fifo size is 16384 it is important since rx priority relies on enough fifo size
    NUM_PACKETS = 1000
    TIMEOUT_S = 10.0

    test_packet = bytes([i & 0xFF for i in range(PACKET_SIZE)])

    dev = open_ftdi()

    stale = drain_rx(dev)
    if stale:
        print("Drained stale RX bytes:", len(stale))

    total_tx = 0
    total_rx = 0

    t_start = time.perf_counter()

    for i in range(NUM_PACKETS):
        dev.write(test_packet)
        total_tx += PACKET_SIZE

        rx = b""
        t0 = time.perf_counter()

        while len(rx) < PACKET_SIZE:
            chunk = dev.read(PACKET_SIZE - len(rx))

            if chunk:
                rx += chunk
            else:
                if time.perf_counter() - t0 > TIMEOUT_S:
                    print("Timeout at packet", i)
                    break
                time.sleep(0.001)

        if rx != test_packet:
            print("Data mismatch at packet", i)
            print("Expected:", test_packet[:32])
            print("Received:", rx[:32])
            break

        total_rx += len(rx)

    t_end = time.perf_counter()
    dev.close()

    elapsed = t_end - t_start

    print()
    print("TX bytes:", total_tx)
    print("RX bytes:", total_rx)
    print("Elapsed:", elapsed, "s")
    print("Echo throughput:", total_rx / elapsed / 1_000_000, "MB/s")
    print("Echo throughput:", total_rx / elapsed / (1024 * 1024), "MiB/s")


if TEST_TX_ONLY == 1:

    START_COMMAND = b"S"

    PACKET_SIZE = 1024
    NUM_PACKETS = 100
    TIMEOUT_S = 1.0

    dev = open_ftdi()

    stale = drain_rx(dev)
    if stale:
        print("Drained stale RX bytes:", len(stale))

    print("Sending burst test start command...")
    dev.write(START_COMMAND)

    total_rx = 0
    error_count = 0
    packet_intervals_ms = []

    t_start = time.perf_counter()
    last_packet_time = t_start

    for packet_idx in range(NUM_PACKETS):

        packet = b""
        t0 = time.perf_counter()

        while len(packet) < PACKET_SIZE:

            chunk = dev.read(PACKET_SIZE - len(packet))

            if chunk:
                packet += chunk
            else:
                if (time.perf_counter() - t0) > TIMEOUT_S:
                    print("Timeout at packet", packet_idx)
                    break

                time.sleep(0.0001)

        now = time.perf_counter()

        if len(packet) != PACKET_SIZE:
            print("Incomplete packet:", packet_idx, "len:", len(packet))
            break

        dt_ms = (now - last_packet_time) * 1000.0
        last_packet_time = now
        packet_intervals_ms.append(dt_ms)

        # Since FPGA resets stream_counter for each packet,
        # each packet should be 00 01 02 ... FF 00 01 ...
        expected_byte = 0

        for i, b in enumerate(packet):

            if b != expected_byte:
                error_count += 1

                if error_count <= 20:
                    print(
                        f"DATA ERROR packet {packet_idx}, "
                        f"byte {i}, "
                        f"expected 0x{expected_byte:02X}, "
                        f"got 0x{b:02X}"
                    )

                expected_byte = (b + 1) & 0xFF
            else:
                expected_byte = (expected_byte + 1) & 0xFF

        total_rx += len(packet)

        print(
            f"Packet {packet_idx}: "
            f"{len(packet)} bytes, "
            f"dt = {dt_ms:.3f} ms"
        )

    t_end = time.perf_counter()
    dev.close()

    elapsed = t_end - t_start
    packets_received = total_rx // PACKET_SIZE

    print()
    print("Packets received:", packets_received)
    print("Received bytes:", total_rx)
    print("Elapsed:", elapsed, "s")

    print(
        "Throughput including gaps:",
        total_rx / elapsed / 1_000_000,
        "MB/s"
    )

    print(
        "Throughput including gaps:",
        total_rx / elapsed / (1024 * 1024),
        "MiB/s"
    )

    if packet_intervals_ms:
        print()
        print("Packet interval min:", min(packet_intervals_ms), "ms")
        print("Packet interval max:", max(packet_intervals_ms), "ms")
        print(
            "Packet interval avg:",
            sum(packet_intervals_ms) / len(packet_intervals_ms),
            "ms"
        )

    if error_count == 0:
        print()
        print("DATA CHECK PASSED")
    else:
        print()
        print("DATA CHECK FAILED")
        print("Total data errors:", error_count)