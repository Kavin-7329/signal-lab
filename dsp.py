"""Core DSP functions for SignalLab (no UI code here, so it is easy to test)."""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.io import wavfile
import io

FILTER_TYPES = ["lowpass", "highpass", "bandpass", "bandstop"]
METHODS = ["FIR (window)", "Butterworth", "Chebyshev I", "Elliptic"]


# ---------- signal generation ----------
def generate_signal(kind, freq, duration, fs, amp=1.0):
    t = np.arange(int(duration * fs)) / fs
    if kind == "sine":
        x = np.sin(2 * np.pi * freq * t)
    elif kind == "square":
        x = signal.square(2 * np.pi * freq * t)
    elif kind == "sawtooth":
        x = signal.sawtooth(2 * np.pi * freq * t)
    elif kind == "chirp":
        x = signal.chirp(t, f0=freq, t1=duration, f1=freq * 10)
    elif kind == "multi-tone":
        x = sum(np.sin(2 * np.pi * f * t) for f in (freq, 2.5 * freq, 4 * freq)) / 3
    else:
        raise ValueError(f"unknown signal kind: {kind}")
    return t, amp * x


def add_noise(x, snr_db, seed=None):
    """Add white Gaussian noise so the result has the requested SNR in dB."""
    rng = np.random.default_rng(seed)
    p_signal = np.mean(x ** 2)
    p_noise = p_signal / (10 ** (snr_db / 10))
    return x + rng.normal(0, np.sqrt(p_noise), size=x.shape)


def snr_db(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.mean(clean ** 2) / max(np.mean(noise ** 2), 1e-20))


# ---------- filters ----------
@dataclass
class Filter:
    kind: str
    method: str
    fs: float
    cutoff: object            # float, or (low, high) for band filters
    order: int
    sos: np.ndarray = None    # IIR
    b: np.ndarray = None      # FIR

    def apply(self, x, zero_phase=True):
        if self.sos is not None:
            return signal.sosfiltfilt(self.sos, x) if zero_phase else signal.sosfilt(self.sos, x)
        return signal.filtfilt(self.b, [1.0], x) if zero_phase else signal.lfilter(self.b, [1.0], x)

    def response(self, n=2048):
        if self.sos is not None:
            w, h = signal.sosfreqz(self.sos, worN=n, fs=self.fs)
        else:
            w, h = signal.freqz(self.b, [1.0], worN=n, fs=self.fs)
        return w, h

    def is_stable(self):
        if self.sos is None:
            return True
        z, p, k = signal.sos2zpk(self.sos)
        return bool(np.all(np.abs(p) < 1))


def design_filter(kind, method, cutoff, fs, order=4, ripple_db=1.0, atten_db=40.0):
    if kind not in FILTER_TYPES:
        raise ValueError("bad filter type")
    nyq = fs / 2
    edges = np.atleast_1d(cutoff).astype(float)
    if kind in ("bandpass", "bandstop"):
        if edges.size != 2 or edges[0] >= edges[1]:
            raise ValueError("band filters need cutoff=(low, high) with low < high")
    elif edges.size != 1:
        raise ValueError("lowpass/highpass need a single cutoff frequency")
    if np.any(edges <= 0) or np.any(edges >= nyq):
        raise ValueError(f"cutoff must be between 0 and Nyquist ({nyq:g} Hz)")

    wn = edges / nyq
    wn = wn if wn.size == 2 else wn[0]
    btype = {"lowpass": "lowpass", "highpass": "highpass",
             "bandpass": "bandpass", "bandstop": "bandstop"}[kind]

    if method == "FIR (window)":
        numtaps = order * 20 + 1            # odd length, valid for every filter type
        pass_zero = {"lowpass": True, "highpass": False,
                     "bandpass": False, "bandstop": True}[kind]
        b = signal.firwin(numtaps, edges, pass_zero=pass_zero, fs=fs, window="hamming")
        return Filter(kind, method, fs, cutoff, order, b=b)

    if method == "Butterworth":
        sos = signal.butter(order, wn, btype=btype, output="sos")
    elif method == "Chebyshev I":
        sos = signal.cheby1(order, ripple_db, wn, btype=btype, output="sos")
    elif method == "Elliptic":
        sos = signal.ellip(order, ripple_db, atten_db, wn, btype=btype, output="sos")
    else:
        raise ValueError("unknown method")
    return Filter(kind, method, fs, cutoff, order, sos=sos)


# ---------- analysis ----------
def spectrum(x, fs):
    """Hann-windowed one-sided magnitude spectrum in dB."""
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    mag = np.abs(X) / (np.sum(w) / 2)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, 20 * np.log10(np.maximum(mag, 1e-12))


def get_spectrogram(x, fs, nperseg=256):
    nperseg = min(nperseg, len(x))
    f, t, S = signal.spectrogram(x, fs, nperseg=nperseg, noverlap=nperseg // 2)
    return f, t, 10 * np.log10(np.maximum(S, 1e-14))


def dominant_frequencies(x, fs, n=3):
    f, db = spectrum(x, fs)
    peaks, _ = signal.find_peaks(db, height=np.max(db) - 30, distance=max(1, len(f) // 200))
    peaks = peaks[np.argsort(db[peaks])[::-1]][:n]
    return [(float(f[i]), float(db[i])) for i in peaks]


# ---------- audio I/O ----------
def read_wav(file_like):
    fs, data = wavfile.read(file_like)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)               # mix down to mono
    peak = np.max(np.abs(data)) or 1.0
    return fs, data / peak


def wav_bytes(x, fs):
    x = np.clip(x / (np.max(np.abs(x)) or 1.0), -1, 1)
    buf = io.BytesIO()
    wavfile.write(buf, int(fs), (x * 32767).astype(np.int16))
    return buf.getvalue()
