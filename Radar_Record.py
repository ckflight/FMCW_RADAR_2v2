# pip3 install pyserial

# pylibftdi needs libftdi use Homebrew and type command in terminal -> brew install libftdi
import pylibftdi as ftdi
import serial.tools.list_ports
import numpy as np
import time
import os
import binascii

ADC_SELECT          = 0             # 0 for ADC DMA, 1 for External ADC MAX1426
RECORD_TIME         = 80            # in sec
TEST_DEVICE         = 1             # 0 STM32F4, 1 STM32H7, 2 FPGA

if ADC_SELECT == 0:
    # 1ms sampling numbers are actual sampling khz freq.

    if TEST_DEVICE == 0:
        # Options: 518KHz(900), 618KHz(900), 778KHz(700), 933KHz(933smp),
        # 1.050MHz(1050smp), 1.556MHz(1556smp), 2.8 MHz(2700smp)
        SAMPLING_FREQUENCY = 1050000 # 58dbfs noise floor
        #SAMPLING_FREQUENCY = 1550000  # 58dbfs noise floor
        #SAMPLING_FREQUENCY = 2600000 # 70dbfs noise floor
        NUMBER_OF_SAMPLES = int(SAMPLING_FREQUENCY / 1000) * 1  # NUMBER_OF_SAMPLES(16bit) = SAMPLING_FREQUENCY * SWEEP_TIME(int)

    if TEST_DEVICE == 1:
        # 16bit Options: 3.72MHz(3720), 3.38MHz(3380)
        # 14bit Options: 4.14MHz(4140), 3.72MHz(3720)
        # 12bit Options: 4.64MHz(4640), 4.14MHz(4140)
        # Oversampling 2 works with highest rates for each bit options.
        SAMPLE_AVERAGING = 2 #1, 2, 4, 8, 16
        SAMPLING_FREQUENCY = int(3720000 / SAMPLE_AVERAGING) # oversampling 2 is enabled
        NUMBER_OF_SAMPLES = int(SAMPLING_FREQUENCY / 1000) * 1  # NUMBER_OF_SAMPLES(16bit) = SAMPLING_FREQUENCY * SWEEP_TIME(int)

    if TEST_DEVICE == 2:
        SAMPLING_FREQUENCY = 500000
        NUMBER_OF_SAMPLES = int(SAMPLING_FREQUENCY / 1000) * 1  # NUMBER_OF_SAMPLES(16bit) = SAMPLING_FREQUENCY * SWEEP_TIME(int)
        BUFFER_LEN = 500*1
else:
    SAMPLING_FREQUENCY  = 400000
    NUMBER_OF_SAMPLES   = 400  # NUMBER_OF_SAMPLE(16bit) = SAMPLING_FREQUENCY * SWEEP_TIME(int)

# VCO range is 0V = 5.1GHz and 10V = 6.3GHz range 1200 max
# 100MHz long range check: usable range 5.2 to 6.1 max and 5.2-5.3 is best 5.3 to 5.8 is good
SWEEP_START         = 5.30e9
SWEEP_BW            = 700e6
TX_POWER_DBM        = 0
SWEEP_TYPE          = 0             # 0 for Sawtooth, 1 for Triangular
USE_PLL             = 1             # 0 for DAC, 1 for PLL
TX_MODE             = 1             # 0 for continuous tx, 1 for on off with tx, 2 for testing when tx off
GAIN                = 10            # 1 to 70 stmf4, 3 to 85 for H7
DATA_LOG            = 0             # 0 for USB transfer, 1 for MicroCard Log
SWEEP_TIME          = 1.0e-3        # in sec, now less than 1ms is working as well
CHECK_MODE          = 0             # 0 ADC_DMA SAMPLING, 1 ADC_DMA USB, 2 MAX1426, 4 FPGA
USB_DATA_TYPE       = 1             # 0-> floating/2 x100 is sent ove usb, 1-> 16bit data is sent
ADC_RESOLUTION      = 16

if DATA_LOG == 1:
    SWEEP_GAP = 1000 * 1.0e-6
else:
    SWEEP_GAP = (2.0 * NUMBER_OF_SAMPLES) * 1.0e-6  # in sec max 4000

    #SWEEP_GAP = 4095 * 1.0e-6
    if SWEEP_GAP >= 4095 * 1.0e-6:
        SWEEP_GAP = 4095 * 1.0e-6

distance = 1
hz_per_m = 0
if SWEEP_BW:
    hz_per_m = int((2* SWEEP_BW * distance) / (3e8 * SWEEP_TIME))


SWEEP_FREQ          = 1 / (SWEEP_TIME + SWEEP_GAP)
RECORD_COUNTER      = int(RECORD_TIME * SWEEP_FREQ)

def Serial_Init():

    isPortConnected = False
    deviceId = ''

    # Find STM32 USB Device Id
    while not isPortConnected:
        ports = serial.tools.list_ports.comports()

        for device_ in ports:

            stm_string = device_.device

            for i in range(len(stm_string) - 8):

                str_to_look = "usbmodem"

                if stm_string[i: i + 8] == str_to_look:

                    device_id_number = stm_string[i + 8: len(stm_string)]
                    deviceId = "/dev/tty." + str_to_look + device_id_number
                    print(device_)
                    isPortConnected = True

                    break

    ser = serial.Serial(
        port=deviceId,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS,
        #timeout=0.01 # can be None
        timeout=None
    )

    # Clear all input data
    ser.flushInput()

    return ser

def Serial_Init_Specific(modem_name):

    deviceId = modem_name
    print(deviceId)

    ser = serial.Serial(
        port=deviceId,
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_TWO,
        bytesize=serial.EIGHTBITS,
        #timeout=0.01 # can be None
        timeout=None
    )

    # Clear all input data
    ser.flushInput()

    return ser

def Device_Init():
    dev_list = ftdi.Driver().list_devices()

    for device_ in dev_list:
        print(device_)

    # Mode is binary b or text t, interface 1 = A, interface 2 = B
    device = ftdi.Device(device_id="FT5US1H5", mode='b', interface_select=ftdi.INTERFACE_A)
    device.open()

    #device.ftdi_fn.ftdi_set_bitmode(0xff, 0x01)
    device.ftdi_fn.ftdi_read_data_set_chunksize(BUFFER_LEN)
    device.ftdi_fn.ftdi_write_data_set_chunksize(BUFFER_LEN)
    #device.ftdi_fn.ftdi_setflowctrl(0x1 << 8)

    device.flush()

    return device

def Configuration_Process():

    isDone = False
    state = 0

    while not isDone:

        # Send start condition and hardware configuration paramaters
        if state == 0:

            ser.write("==".encode('ascii'))

            sw_t = np.uint16(SWEEP_TIME * 1e6)
            sw_t_msb = np.uint8((sw_t >> 8) & 0xFF)
            sw_t_lsb = np.uint8(sw_t & 0xFF)

            ser.write(binascii.hexlify(sw_t_msb))
            ser.write(binascii.hexlify(sw_t_lsb))

            sw_g = np.uint16(SWEEP_GAP * 1e6)
            sw_g_msb = np.uint8((sw_g >> 8) & 0xFF)
            sw_g_lsb = np.uint8(sw_g & 0xFF)

            ser.write(binascii.hexlify(sw_g_msb))
            ser.write(binascii.hexlify(sw_g_lsb))

            rec_t = np.uint8(RECORD_TIME)
            ser.write(binascii.hexlify(rec_t))

            fs = np.uint16(SAMPLING_FREQUENCY / 1e3)
            fs_msb = np.uint8((fs >> 8) & 0xFF)
            fs_lsb = np.uint8(fs & 0xFF)

            ser.write(binascii.hexlify(fs_msb))
            ser.write(binascii.hexlify(fs_lsb))

            num_sample = np.uint16(NUMBER_OF_SAMPLES)
            num_sample_msb = np.uint8((num_sample >> 8) & 0xFF)
            num_sample_lsb = np.uint8(num_sample & 0xFF)

            ser.write(binascii.hexlify(num_sample_msb))
            ser.write(binascii.hexlify(num_sample_lsb))

            sweep_start = np.uint16(SWEEP_START / 1e7)
            sweep_start_msb = np.uint8((sweep_start >> 8) & 0xFF)
            sweep_start_lsb = np.uint8(sweep_start & 0xFF)

            ser.write(binascii.hexlify(sweep_start_msb))
            ser.write(binascii.hexlify(sweep_start_lsb))

            sweep_bw = np.uint16(SWEEP_BW / 1e6)
            sweep_bw_msb = np.uint8((sweep_bw >> 8) & 0xFF)
            sweep_bw_lsb = np.uint8(sweep_bw & 0xFF)

            ser.write(binascii.hexlify(sweep_bw_msb))
            ser.write(binascii.hexlify(sweep_bw_lsb))

            tx_mode = np.uint8(TX_MODE)
            ser.write(binascii.hexlify(tx_mode))

            gain = np.uint8(GAIN)
            ser.write(binascii.hexlify(gain))

            sw_type = np.uint8(SWEEP_TYPE)
            ser.write(binascii.hexlify(sw_type))

            data_log = np.uint8(DATA_LOG)
            ser.write(binascii.hexlify(data_log))

            adc_select = np.uint8(ADC_SELECT)
            ser.write(binascii.hexlify(adc_select))

            use_pll = np.uint8(USE_PLL)
            ser.write(binascii.hexlify(use_pll))

            check_mode = np.uint8(CHECK_MODE)
            ser.write(binascii.hexlify(check_mode))

            usb_data_type = np.uint8(USB_DATA_TYPE)
            ser.write(binascii.hexlify(usb_data_type))

            adc_resolution = np.uint8(ADC_RESOLUTION)
            ser.write(binascii.hexlify(adc_resolution))

            sample_averaging = np.uint8(SAMPLE_AVERAGING)
            ser.write(binascii.hexlify(sample_averaging))

            state = 1
            pass

        # Receive response and tx power from radar
        elif state == 1:

            # Wait until response is received or timeout
            while ser.inWaiting() == 0 and state == 1:
                pass

            if ser.inWaiting() > 0:
                config_read = ser.read(ser.inWaiting())

                if config_read[0:2] == str("==").encode('ascii'):

                    dbm = config_read[2]
                    dbm_voltage = config_read[3]
                    print("dbm_voltage",dbm_voltage)
                    print("dbm", dbm)
                    isDone = True
            pass

    return dbm, dbm_voltage

if TEST_DEVICE == 0:
    ser = Serial_Init_Specific("/dev/cu.usbmodem3158397630341")
    TX_POWER_DBM, TX_POWER_DBM_VOLTAGE = Configuration_Process()

elif TEST_DEVICE == 1:
    ser = Serial_Init_Specific("/dev/tty.usbmodem3878386530331")
    TX_POWER_DBM, TX_POWER_DBM_VOLTAGE = Configuration_Process()

elif TEST_DEVICE == 2:
    ser = Serial_Init_Specific("/dev/cu.usbmodem3158397630341")
    TX_POWER_DBM, TX_POWER_DBM_VOLTAGE = Configuration_Process()
    ser.close()#!!!!!!!!!!!!!!!!!!!

    device = Device_Init()

if DATA_LOG == 0:

    if os.path.exists("Radar_Records/data_record.txt"):
        os.remove("Radar_Records/data_record.txt")
    data_record_file = open("Radar_Records/data_record.txt", "w")

    rx_counter = 0
    isRecordDone = 0

    data_record_file.write(str(RECORD_COUNTER))
    data_record_file.write("\r\n")
    data_record_file.write(str(RECORD_TIME))
    data_record_file.write("\r\n")
    data_record_file.write(str(int(SWEEP_TIME * 1000000)))
    data_record_file.write("\r\n")
    data_record_file.write(str(int(SWEEP_GAP * 1000000)))
    data_record_file.write("\r\n")
    data_record_file.write(str(int(SWEEP_START)))
    data_record_file.write("\r\n")
    data_record_file.write(str(int(SWEEP_BW)))
    data_record_file.write("\r\n")
    data_record_file.write(str(SAMPLING_FREQUENCY))
    data_record_file.write("\r\n")
    data_record_file.write(str(NUMBER_OF_SAMPLES))
    data_record_file.write("\r\n")
    data_record_file.write(str(TX_MODE))
    data_record_file.write("\r\n")
    data_record_file.write(str(TX_POWER_DBM))
    data_record_file.write("\r\n")
    data_record_file.write(str(TX_POWER_DBM_VOLTAGE))
    data_record_file.write("\r\n")
    data_record_file.write(str(hz_per_m))
    data_record_file.write("\r\n")
    data_record_file.write(str(DATA_LOG))
    data_record_file.write("\r\n")
    data_record_file.write(str(ADC_SELECT))
    data_record_file.write("\r\n")
    data_record_file.write(str(USB_DATA_TYPE))
    data_record_file.write("\r\n")
    data_record_file.write(str(ADC_RESOLUTION))
    data_record_file.write("\r\n")

    start_time = time.time()

    while not isRecordDone:

        if TEST_DEVICE == 2:

            calculation_start = time.time()

            time.sleep(0.004)
            rx_data = device.read(NUMBER_OF_SAMPLES * 2)

            if len(rx_data) != 0:

                rx_data_str = str(binascii.b2a_hex(rx_data))  # takes hex values and stores them as string working perfectly

                rx_data_length = len(rx_data_str)
                print("RX:")
                print(rx_data_length)
                print(rx_data_str)

                # b'.....' so remove these 3 bytes
                data_record_file.write(rx_data_str[2:len(rx_data_str) - 1])
                data_record_file.write("\r\n")

        else:
            # Test mode logic is not working with adc samples because hex values are generating any type of character
            # As a result best way of doing is just sending sampling data in 2 bytes without any start stop bytes
            # Now i am sending 0 to 65536 values with one increment at a time to test and it works. Keep it simple!!!
            read_size = ser.inWaiting()

            calculation_start = time.time()

            # if fix byte is read first and last transfers will be ignored
            if(USB_DATA_TYPE == 0):
                rx_data = ser.read(NUMBER_OF_SAMPLES)  # 8 bit scaled float NUMBER_OF_SAMPLES
            elif(USB_DATA_TYPE == 1):
                rx_data = ser.read(NUMBER_OF_SAMPLES * 2)  # 12, 16 bit NUMBER_OF_SAMPLES = 8 bit 2*NUMBER_OF_SAMPLES

            rx_data_str = str(binascii.b2a_hex(rx_data))  # takes hex values and stores them as string working perfectly
            rx_data_length = len(rx_data_str)

            # b'.....' so remove these 3 bytes
            data_record_file.write(rx_data_str[2:len(rx_data_str) - 1])
            data_record_file.write("\r\n")

        rx_counter += 1;
        if rx_counter == RECORD_COUNTER:
            isRecordDone = 1
            done_time = time.time() - start_time
            print("Completed in ", str(round(done_time, 1)), "second.")

        calculation_done = time.time() - calculation_start

        print("Calculation time: ", str(round(calculation_done, 4)), "   rx counter: ", str(rx_counter))
        #print(rx_counter)

    print("TX Power: ", str(TX_POWER_DBM))
    print("TX Power: ", str(TX_POWER_DBM_VOLTAGE / 100.0))
    print("Meters per KHz: ",str(hz_per_m))
    exit(1)









