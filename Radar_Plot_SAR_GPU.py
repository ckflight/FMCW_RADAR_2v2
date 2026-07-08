import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp_cpu

try:
    import cupy as cp
except ImportError as e:
    raise SystemExit("Install CuPy first: pip install cupy-cuda12x") from e


# ============================================================
# FILE
# ============================================================
TXT_FILE = "Radar_Records/radar2v2_horn_48kHz_2024_04_09_16_41_58_parking_lot_sar.txt"


# ============================================================
# USER SETTINGS
# ============================================================
ENABLE_SPEED_SEARCH = True

SPEED_SEARCH_MIN = 1.0
SPEED_SEARCH_MAX = 2.2
SPEED_SEARCH_STEPS = 13

speed = 1.5

sample_increment = 1
remove_first_sweeps = 0
CUT_END_SWEEPS = 0

dynamic_range = 45

# Important for old TXT radar:
# search padding 1 avoids OOM, final padding 2 gives nicer display
cross_range_padding_search = 1
cross_range_padding_final = 2

remove_leakage_bins = 10
remove_static_clutter = True
use_rvp = True

ENABLE_AUTOFOCUS = True
PGA_ITERATIONS = 8

interp_kind = "cubic"

c = 299792458.0


# ============================================================
# HELPERS
# ============================================================
def clear_gpu():
    gc.collect()
    try:
        cp.fft.config.get_plan_cache().clear()
    except Exception:
        pass

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def hilbert_rvp_gpu(x, fs, kr_slope):
    y = cp.fft.fft(x, axis=-1)
    y[:, : y.shape[1] // 2 + 1] = 0

    f = cp.linspace(-fs / 2, fs / 2, y.shape[1], dtype=cp.float32)
    y *= cp.exp(-1j * cp.pi * f**2 / kr_slope).astype(cp.complex64)

    return cp.fft.ifft(y, axis=-1).astype(cp.complex64)


def pga_autofocus(img, iterations=8):
    img = img.astype(cp.complex64, copy=True)

    n_az, n_rg = img.shape

    power_rg = cp.max(cp.abs(img), axis=0)
    th = cp.percentile(power_rg, 92)
    rg = cp.where(power_rg > th)[0]

    if int(rg.size) < 8:
        return img

    center = n_az // 2
    half = int(n_az * 0.30)

    az0 = max(0, center - half)
    az1 = min(n_az, center + half)

    for _ in range(iterations):
        sub = img[az0:az1, rg]

        peak = cp.argmax(cp.abs(sub), axis=0)
        aligned = cp.zeros_like(sub)

        ref = sub.shape[0] // 2

        for k in range(sub.shape[1]):
            shift = ref - int(peak[k].get())
            aligned[:, k] = cp.roll(sub[:, k], shift)

        spec = cp.fft.fftshift(cp.fft.fft(aligned, axis=0), axes=0)

        d = cp.conj(spec[:-1, :]) * spec[1:, :]
        grad = cp.angle(cp.sum(d, axis=1))

        phase = cp.cumsum(cp.concatenate([cp.zeros(1, dtype=grad.dtype), grad]))
        phase = cp.unwrap(phase)
        phase -= cp.mean(phase)

        full_phase = cp.interp(
            cp.arange(n_az, dtype=cp.float32),
            cp.linspace(az0, az1 - 1, phase.size, dtype=cp.float32),
            phase.astype(cp.float32),
        )

        img *= cp.exp(-1j * full_phase).astype(cp.complex64)[:, None]

        del sub, peak, aligned, spec, d, grad, phase, full_phase
        clear_gpu()

    return img


def focus_metric(img):
    mag = cp.abs(img)

    a0 = img.shape[0] // 10
    a1 = img.shape[0] - a0
    r0 = max(5, img.shape[1] // 40)
    r1 = img.shape[1] - img.shape[1] // 20

    mag = mag[a0:a1, r0:r1]

    p = mag**2
    p = p / (cp.mean(p) + 1e-12)

    contrast = cp.mean(p**2)
    entropy = -cp.sum((p / cp.sum(p)) * cp.log(p / cp.sum(p) + 1e-12))

    return float((contrast / entropy).get())


def form_image_original_quality(
    data1,
    speed,
    fs,
    tsweep,
    pri,
    bw,
    fc,
    padding_factor=1,
    make_db=True,
    do_autofocus=True,
):
    start_time = time.time()

    sweep_samples = data1.shape[1]
    delta_crange = pri * speed

    data = cp.asarray(data1, dtype=cp.float32)

    # DC remove per sweep
    data = data - cp.mean(data, axis=1, keepdims=True)

    # fast-time window
    wf = cp.hanning(sweep_samples).astype(cp.float32)
    data = data * wf[None, :]

    # analytic signal + RVP
    if use_rvp:
        data = hilbert_rvp_gpu(data, fs, bw / tsweep)

    # static clutter remove
    if remove_static_clutter:
        data = data - cp.mean(data, axis=0, keepdims=True)

    # cross-range zero padding
    if padding_factor > 1:
        zpad = int((padding_factor - 1) * data.shape[0])
        data = cp.pad(
            data,
            ((zpad // 2, zpad - zpad // 2), (0, 0)),
            mode="constant",
        )

    # azimuth window
    ws = cp.hanning(data.shape[0]).astype(cp.float32)[:, None]
    data = data * ws

    # k axes
    kx = cp.linspace(
        -cp.pi / delta_crange,
        cp.pi / delta_crange,
        data.shape[0],
        dtype=cp.float32,
    )

    kr = cp.linspace(
        (4.0 * cp.pi / c) * (fc - bw / 2.0),
        (4.0 * cp.pi / c) * (fc + bw / 2.0),
        sweep_samples,
        dtype=cp.float32,
    )

    # azimuth FFT
    cfft = cp.fft.fftshift(cp.fft.fft(data, axis=0), axes=0).astype(cp.complex64)

    if remove_leakage_bins > 0:
        cfft[:, :remove_leakage_bins] = 0

    kr_delta = kr[1] - kr[0]
    kx_max = cp.max(cp.abs(kx))

    ky0 = cp.sqrt(cp.maximum(kr[0] ** 2 - kx_max ** 2, cp.float32(1e-12)))
    ky_delta = 1.5 * kr_delta

    ky_even = cp.arange(
        float(ky0.get()),
        float(kr[-1].get()),
        float(ky_delta.get()),
        dtype=cp.float32,
    )

    st = cp.zeros((cfft.shape[0], len(ky_even)), dtype=cp.complex64)

    kr_np = cp.asnumpy(kr)
    kx_np = cp.asnumpy(kx)
    ky_even_np = cp.asnumpy(ky_even)

    print(
        f"speed={speed:.3f} | "
        f"padding={padding_factor} | "
        f"Stolt size: {st.shape[0]} x {st.shape[1]}"
    )

    # ========================================================
    # ORIGINAL-QUALITY CUBIC STOLT INTERPOLATION
    # ========================================================
    for i in range(len(kx_np)):
        val = kr_np**2 - kx_np[i] ** 2
        valid = val > 0

        if np.count_nonzero(valid) < 4:
            continue

        ky = np.sqrt(val[valid])
        row = cp.asnumpy(cfft[i, valid])

        ky_unique, unique_idx = np.unique(ky, return_index=True)
        row_unique = row[unique_idx]

        if len(ky_unique) < 4:
            continue

        ci = interp_cpu.interp1d(
            ky_unique,
            row_unique,
            kind=interp_kind,
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )

        st[i, :] = cp.asarray(ci(ky_even_np), dtype=cp.complex64)

    del data, cfft, kr, kx, ky_even
    clear_gpu()

    # 2D window, no huge outer temporary
    wx = cp.sqrt(cp.hanning(st.shape[0]).astype(cp.float32))
    wy = cp.sqrt(cp.hanning(st.shape[1]).astype(cp.float32))

    st *= wx[:, None]
    st *= wy[None, :]

    img = cp.fft.ifft2(st).astype(cp.complex64)
    img = cp.fft.fftshift(img, axes=0)

    del st, wx, wy
    clear_gpu()

    if do_autofocus:
        img = pga_autofocus(img, PGA_ITERATIONS)

    metric = focus_metric(img)

    mid = img.shape[0] // 2
    img_combined = cp.vstack((img[mid:, :], img[:mid, :]))

    del img
    clear_gpu()

    range_max = c * (fs / 2.0) * tsweep / (2.0 * bw)
    crange = delta_crange * img_combined.shape[0]

    if make_db:
        img_db_gpu = 20 * cp.log10(cp.abs(img_combined) + 1e-12)
        img_db = cp.asnumpy(img_db_gpu)

        del img_combined, img_db_gpu
        clear_gpu()

        return img_db, range_max, crange, metric, time.time() - start_time

    del img_combined
    clear_gpu()

    return None, range_max, crange, metric, time.time() - start_time


# ============================================================
# READ TXT HEADER
# ============================================================
record_file = open(TXT_FILE, "r")

RECORD_COUNTER = int(record_file.readline().strip())
print("Record Counter:", RECORD_COUNTER)

RECORD_TIME = int(record_file.readline().strip())
print("Record Time:", RECORD_TIME, "sec.")

SWEEP_TIME = int(record_file.readline().strip()) / 1e6
print("Sweep Time:", SWEEP_TIME, "sec.")

SWEEP_DELAY = int(record_file.readline().strip()) / 1e6
print("Sweep Delay:", SWEEP_DELAY, "sec.")

SWEEP_START = int(record_file.readline().strip())
print("Sweep Start:", SWEEP_START, "Hz")

SWEEP_BW = int(record_file.readline().strip())
print("Sweep BW:", SWEEP_BW, "Hz")

SAMPLING_FREQUENCY = int(record_file.readline().strip())
print("Sampling Frequency:", SAMPLING_FREQUENCY, "Hz")

NUMBER_OF_SAMPLES = int(record_file.readline().strip())
print("Samples per sweep:", NUMBER_OF_SAMPLES)

TX_MODE = int(record_file.readline().strip())
print("Tx Mode:", TX_MODE)

TX_POWER_DBM = int(record_file.readline().strip())
print("Tx Power:", TX_POWER_DBM, "dBm")

TX_POWER_DBM_VOLTAGE = int(record_file.readline().strip())
print("Tx Power Voltage:", TX_POWER_DBM_VOLTAGE / 100.0, "V")

hz_per_m = int(record_file.readline().strip())
print("Hz per m:", hz_per_m)

DATA_LOG = int(record_file.readline().strip())
print("Data Log:", DATA_LOG)

USB_DATA_TYPE = int(record_file.readline().strip())
print("USB Data Type:", USB_DATA_TYPE)

ADC_RESOLUTION = int(record_file.readline().strip())
print("ADC Resolution:", ADC_RESOLUTION)

PHASE_DISTANCE = int(record_file.readline().strip())
print("Phase Distance:", PHASE_DISTANCE)

CHIRP_NUMBER = record_file.readline().strip()
print("Chirp Number:", CHIRP_NUMBER)

RECORD_DATE = record_file.readline().strip()
print("Date:", RECORD_DATE)


# ============================================================
# LOAD RAW SWEEPS
# ============================================================
usable_records = int(RECORD_COUNTER / sample_increment)
data1 = np.zeros((usable_records, NUMBER_OF_SAMPLES), dtype=np.float32)

sample_counter = 0
data_counter = 0

while data_counter < RECORD_COUNTER:
    sample_line = record_file.readline()

    if not sample_line:
        break

    if data_counter % sample_increment != 0:
        data_counter += 1
        continue

    samples_hex = bytes.fromhex(sample_line)
    length_line = len(samples_hex)

    if USB_DATA_TYPE == 0:
        samples = np.frombuffer(samples_hex, dtype=np.uint8).astype(np.float32)
        samples = (samples / 150.0) * 3.3
        data1[sample_counter, :] = samples[:NUMBER_OF_SAMPLES]

    elif USB_DATA_TYPE == 1:
        index = 0
        append_counter = 0

        while index + 1 < length_line and append_counter < NUMBER_OF_SAMPLES:
            current_sample_16bit = (
                ((samples_hex[index] & 0xFF) << 8)
                | (samples_hex[index + 1] & 0xFF)
            )

            current_sample_float = (
                current_sample_16bit / (2 ** ADC_RESOLUTION)
            ) * 3.3

            data1[sample_counter, append_counter] = current_sample_float

            index += 2
            append_counter += 1

    sample_counter += 1
    data_counter += 1

record_file.close()

data1 = data1[:sample_counter, :]

if remove_first_sweeps > 0:
    data1 = data1[remove_first_sweeps:, :]

if CUT_END_SWEEPS > 0:
    data1 = data1[:-CUT_END_SWEEPS, :]

print("Loaded sweeps:", data1.shape)


# ============================================================
# PARAMETERS
# ============================================================
fs = SAMPLING_FREQUENCY
tsweep = SWEEP_TIME
bw = SWEEP_BW
fc = SWEEP_START + bw / 2.0
lam = c / fc

tdelay = SWEEP_DELAY + ((SWEEP_TIME + SWEEP_DELAY) * (sample_increment - 1))
pri = tsweep + tdelay

print("----- APERTURE -----")
print(f"PRI          : {pri*1e6:.3f} us")
print(f"PRF          : {1.0/pri:.2f} Hz")
print(f"carrier      : {fc/1e9:.3f} GHz")
print(f"lambda       : {lam*1000:.2f} mm")


# ============================================================
# SPEED SEARCH
# ============================================================
if ENABLE_SPEED_SEARCH:
    print("----- SPEED SEARCH, CUBIC STOLT + PGA -----")

    speeds = np.linspace(SPEED_SEARCH_MIN, SPEED_SEARCH_MAX, SPEED_SEARCH_STEPS)

    metrics = []
    best_speed = speeds[0]
    best_metric = -1e99

    for s in speeds:
        try:
            _, _, _, metric, dt = form_image_original_quality(
                data1=data1,
                speed=s,
                fs=fs,
                tsweep=tsweep,
                pri=pri,
                bw=bw,
                fc=fc,
                padding_factor=cross_range_padding_search,
                make_db=False,
                do_autofocus=ENABLE_AUTOFOCUS,
            )

        except Exception as e:
            print(f"speed={s:.3f} failed: {e}")
            metric = -1e99
            dt = 0.0

        clear_gpu()
        metrics.append(metric)

        print(
            f"speed={s:.3f} m/s | "
            f"delta={s*pri*1000:.3f} mm/sweep | "
            f"metric={metric:.6f} | "
            f"time={dt:.2f}s"
        )

        if metric > best_metric:
            best_metric = metric
            best_speed = s

    speed = best_speed

    print(f"BEST SPEED = {speed:.3f} m/s")

    plt.figure(figsize=(7, 4))
    plt.plot(speeds, metrics, marker="o")
    plt.grid(True)
    plt.xlabel("Speed [m/s]")
    plt.ylabel("Focus metric")
    plt.title("Speed Search, Cubic Stolt + PGA")
    plt.tight_layout()
    plt.show()


# ============================================================
# FINAL IMAGE
# ============================================================
print("----- FINAL IMAGE -----")
print(f"speed        : {speed:.3f} m/s")
print(f"delta_crange : {speed*pri*1000:.3f} mm/sweep")
print(f"delta/lambda : {(speed*pri)/lam:.4f}")

img_db, range_max, crange, metric, dt = form_image_original_quality(
    data1=data1,
    speed=speed,
    fs=fs,
    tsweep=tsweep,
    pri=pri,
    bw=bw,
    fc=fc,
    padding_factor=cross_range_padding_final,
    make_db=True,
    do_autofocus=ENABLE_AUTOFOCUS,
)

print(f"final metric : {metric:.6f}")
print(f"GPU pipeline time: {dt:.3f} s")


# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(10, 7))
plt.title(f"SAR Image | speed={speed:.3f} m/s | Cubic Stolt + PGA")
plt.imshow(
    img_db,
    aspect="auto",
    interpolation="none",
    extent=[0, range_max, -crange / 2.0, crange / 2.0],
    origin="lower",
)

m = np.max(img_db)
plt.clim(m - dynamic_range, m)

plt.colorbar(label="dB")
plt.xlabel("Range [m]")
plt.ylabel("Cross-range [m]")
plt.tight_layout()
plt.show()