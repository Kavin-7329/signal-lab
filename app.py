"""SignalLab: Streamlit UI.  Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import dsp

st.set_page_config(page_title="SignalLab", layout="wide")
st.title("SignalLab: signal generator and digital filter designer")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("1. Input signal")
    source = st.radio("Source", ["Generate", "Upload WAV"], horizontal=True)
    if source == "Generate":
        fs = st.select_slider("Sampling rate (Hz)", [4000, 8000, 16000, 44100], value=8000)
        kind = st.selectbox("Waveform", ["multi-tone", "sine", "square", "sawtooth", "chirp"])
        freq = st.slider("Base frequency (Hz)", 50, 1000, 200, 10)
        duration = st.slider("Duration (s)", 0.2, 3.0, 1.0, 0.1)
        t, clean = dsp.generate_signal(kind, freq, duration, fs)
    else:
        up = st.file_uploader("WAV file", type=["wav"])
        if up is None:
            st.info("Upload a WAV file, or switch to Generate.")
            st.stop()
        fs, clean = dsp.read_wav(up)
        clean = clean[: fs * 5]                    # first 5 seconds keeps the app fast
        t = np.arange(len(clean)) / fs

    st.header("2. Noise")
    snr = st.slider("SNR (dB)", 0, 40, 10)
    noisy = dsp.add_noise(clean, snr, seed=1) if st.checkbox("Add noise", True) else clean

    st.header("3. Filter")
    ftype = st.selectbox("Type", dsp.FILTER_TYPES)
    method = st.selectbox("Method", dsp.METHODS)
    nyq = fs / 2
    if ftype in ("bandpass", "bandstop"):
        lo, hi = st.slider("Band edges (Hz)", 10.0, float(nyq - 10), (float(nyq * 0.05), float(nyq * 0.25)))
        cutoff = (lo, hi)
    else:
        cutoff = st.slider("Cutoff (Hz)", 10.0, float(nyq - 10), float(nyq * 0.15))
    order = st.slider("Order", 1, 10, 4)
    ripple = st.slider("Passband ripple (dB)", 0.1, 3.0, 1.0) if method in ("Chebyshev I", "Elliptic") else 1.0
    zero_phase = st.checkbox("Zero-phase (filtfilt)", True)

# ---------------- processing ----------------
try:
    flt = dsp.design_filter(ftype, method, cutoff, fs, order, ripple)
except ValueError as e:
    st.error(str(e))
    st.stop()
if not flt.is_stable():
    st.warning("This filter is unstable. Lower the order or move the cutoff away from 0 and Nyquist.")
    st.stop()
out = flt.apply(noisy, zero_phase)

# ---------------- metrics ----------------
c1, c2, c3 = st.columns(3)
c1.metric("SNR before filtering", f"{dsp.snr_db(clean, noisy):.1f} dB")
c2.metric("SNR after filtering", f"{dsp.snr_db(clean, out):.1f} dB",
          f"{dsp.snr_db(clean, out) - dsp.snr_db(clean, noisy):+.1f} dB")
peaks = dsp.dominant_frequencies(out, fs)
c3.metric("Strongest output tone", f"{peaks[0][0]:.0f} Hz" if peaks else "none")

# ---------------- plots ----------------
n = min(len(t), int(fs * 0.03))                    # show the first 30 ms
fig, ax = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

ax[0, 0].plot(t[:n] * 1000, noisy[:n], label="input", alpha=0.6)
ax[0, 0].plot(t[:n] * 1000, out[:n], label="filtered", lw=1.8)
ax[0, 0].set(title="Time domain (first 30 ms)", xlabel="ms", ylabel="amplitude")
ax[0, 0].legend()

f1, s1 = dsp.spectrum(noisy, fs)
f2, s2 = dsp.spectrum(out, fs)
ax[0, 1].plot(f1, s1, alpha=0.6, label="input")
ax[0, 1].plot(f2, s2, label="filtered")
ax[0, 1].set(title="Spectrum", xlabel="Hz", ylabel="dB", ylim=(-100, 5))
ax[0, 1].legend()

w, h = flt.response()
ax[1, 0].plot(w, 20 * np.log10(np.maximum(np.abs(h), 1e-6)))
ax[1, 0].set(title="Filter magnitude response", xlabel="Hz", ylabel="dB", ylim=(-100, 5))
ax[1, 0].grid(alpha=0.3)
for e in np.atleast_1d(cutoff):
    ax[1, 0].axvline(e, color="r", ls="--", lw=0.8)

f, tt, S = dsp.get_spectrogram(out, fs)
ax[1, 1].pcolormesh(tt, f, S, shading="auto")
ax[1, 1].set(title="Spectrogram of filtered signal", xlabel="s", ylabel="Hz")

st.pyplot(fig)
st.download_button("Download filtered WAV", dsp.wav_bytes(out, fs), "filtered.wav", "audio/wav")
st.audio(dsp.wav_bytes(out, fs), format="audio/wav")
