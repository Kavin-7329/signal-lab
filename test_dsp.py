import numpy as np
import pytest
import dsp

FS = 8000


def tone_gain(flt, f):
    t = np.arange(FS) / FS
    x = np.sin(2 * np.pi * f * t)
    y = flt.apply(x, zero_phase=True)
    return np.sqrt(np.mean(y[500:-500] ** 2)) / np.sqrt(np.mean(x[500:-500] ** 2))


@pytest.mark.parametrize("method", dsp.METHODS)
def test_lowpass_passes_low_blocks_high(method):
    f = dsp.design_filter("lowpass", method, 500, FS, order=6)
    assert tone_gain(f, 100) > 0.8
    assert tone_gain(f, 2000) < 0.05


@pytest.mark.parametrize("method", dsp.METHODS)
def test_highpass_and_bands(method):
    hp = dsp.design_filter("highpass", method, 1000, FS, order=6)
    assert tone_gain(hp, 3000) > 0.8 and tone_gain(hp, 100) < 0.05
    bp = dsp.design_filter("bandpass", method, (800, 1200), FS, order=6)
    assert tone_gain(bp, 1000) > 0.7 and tone_gain(bp, 100) < 0.1
    bs = dsp.design_filter("bandstop", method, (800, 1200), FS, order=6)
    assert tone_gain(bs, 1000) < 0.2 and tone_gain(bs, 100) > 0.75


def test_filter_improves_snr():
    t, clean = dsp.generate_signal("sine", 200, 1.0, FS)
    noisy = dsp.add_noise(clean, 5, seed=0)
    out = dsp.design_filter("lowpass", "Butterworth", 400, FS, 6).apply(noisy)
    assert dsp.snr_db(clean, out) > dsp.snr_db(clean, noisy) + 5


def test_noise_snr_is_accurate():
    t, clean = dsp.generate_signal("sine", 200, 3.0, FS)
    assert abs(dsp.snr_db(clean, dsp.add_noise(clean, 12, seed=2)) - 12) < 0.5


def test_spectrum_peak_location():
    t, x = dsp.generate_signal("sine", 440, 1.0, FS)
    f, db = dsp.spectrum(x, FS)
    assert abs(f[np.argmax(db)] - 440) < 2


def test_bad_inputs():
    with pytest.raises(ValueError):
        dsp.design_filter("lowpass", "Butterworth", 5000, FS)      # above Nyquist
    with pytest.raises(ValueError):
        dsp.design_filter("bandpass", "Butterworth", (1200, 800), FS)


def test_wav_round_trip():
    t, x = dsp.generate_signal("sine", 300, 0.5, FS)
    fs2, y = dsp.read_wav(__import__("io").BytesIO(dsp.wav_bytes(x, FS)))
    assert fs2 == FS and np.corrcoef(x, y)[0, 1] > 0.999
