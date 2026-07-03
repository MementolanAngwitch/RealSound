"""
string_analysis.py — the ANALYSIS engine for the string track.

Mirrors brass.py's separation of concerns: this half answers *why the physics
works*. It contains only measurement / diagnostic tools — spectra, pitch,
partial extraction, inharmonicity fitting, decay slopes. It is pure DSP math:
no file I/O, no audio playback (§5 of PROJECT_GUIDE — the DSP/analysis core
knows nothing about UI or I/O). The synthesis engine (string_synth.py) imports
the DSP nodes from here so the two stay in sync.

Physics references (§x.y) point at PROJECT_GUIDE.md.
"""
import numpy as np
from scipy.signal import butter, freqz, find_peaks, fftconvolve, lfilter

FS = 44100

# ==========================================================================
# DSP nodes (shared with the synthesis engine)
# ==========================================================================
class DelayLine:
    """Ring buffer. read(d) = value written d samples ago (d ≥ 1).
    read_frac does linear-interpolated fractional delay — mild HF loss,
    conveniently stabilising (§6.1, §6.3)."""
    def __init__(self, size):
        self.buf = np.zeros(int(size) + 4)
        self.n = len(self.buf)
        self.i = 0
    def write(self, x):
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.n
    def read(self, d):
        return self.buf[(self.i - int(d)) % self.n]
    def read_frac(self, d):
        m = int(d); e = d - m
        return (1 - e) * self.read(m) + e * self.read(m + 1)

class OnePole:
    """First-order filter run sample-by-sample with EXPLICIT state (x1, y1).
    This is what scipy's zi vector encodes (§6.2) — carrying it by hand makes
    the state impossible to forget and is ~20× faster per sample than lfilter."""
    def __init__(self, b, a, gain=1.0):
        self.b0, self.b1 = b[0] * gain, b[1] * gain
        self.a1 = a[1]
        self.x1 = 0.0
        self.y1 = 0.0
    def tick(self, x):
        y = self.b0 * x + self.b1 * self.x1 - self.a1 * self.y1
        self.x1, self.y1 = x, y
        return y

class AllpassChain:
    """M cascaded 1st-order all-passes  H(z) = (a + z⁻¹)/(1 + a z⁻¹).
    |H| = 1 at every frequency (lossless — passivity preserved, §6.3) but the
    DELAY is frequency-dependent. With a < 0, high frequencies get LESS delay
    → high partials arrive early → sharp: exactly bending-stiffness dispersion."""
    def __init__(self, a, M):
        self.a, self.M = a, M
        self.x1 = np.zeros(M)
        self.y1 = np.zeros(M)
    def tick(self, x):
        a = self.a
        for k in range(self.M):
            y = a * x + self.x1[k] - a * self.y1[k]
            self.x1[k], self.y1[k] = x, y
            x = y
        return x
    def phase_delay(self, f0):
        return self.M * phase_delay([self.a, 1.0], [1.0, self.a], f0)

class FracDelay:
    """Standalone linear-interp fractional delay 0 ≤ d < 3 for loop tuning."""
    def __init__(self, d):
        self.d = d
        self.h = np.zeros(4)
    def tick(self, x):
        self.h[1:] = self.h[:-1]
        self.h[0] = x
        m = int(self.d); e = self.d - m
        return (1 - e) * self.h[m] + e * self.h[m + 1]

def loss_filter(fc, gain):
    """Frequency-dependent boundary loss (§2.4): 1st-order Butterworth LP.
    gain < 1 handles the flat part of the loss; the LP shape strips top end
    a little more each round trip → 'warm' decay. Returns (node, b, a) so the
    caller can compensate the loop for the filter's phase delay."""
    b, a = butter(1, fc / (FS / 2))
    return OnePole(b, a, gain), b, a

# ==========================================================================
# Measurement / diagnostics — the "why it works" layer
# ==========================================================================
def spectrum_db(y, pad=4):
    """Windowed, zero-padded magnitude spectrum in dB re: max."""
    y = np.asarray(y, float)
    y = y - y.mean()                      # crude DC block for analysis only
    w = np.hanning(len(y))
    Y = np.abs(np.fft.rfft(y * w, n=pad * len(y)))
    f = np.fft.rfftfreq(pad * len(y), 1 / FS)
    return f, 20 * np.log10(Y / (Y.max() + 1e-300) + 1e-12)

def partials(y, f0_guess, fmax=6000, floor_db=-75, pad=4):
    """Find spectral peaks (parabolic-refined) → (freqs, mags in dB)."""
    f, db = spectrum_db(y, pad)
    sel = f <= fmax
    f, db = f[sel], db[sel]
    binhz = f[1] - f[0]
    idx, _ = find_peaks(db, height=floor_db,
                        distance=max(1, int(0.5 * f0_guess / binhz)),
                        prominence=6)
    freqs, mags = [], []
    for i in idx:
        if 0 < i < len(db) - 1:
            a, b, c = db[i - 1], db[i], db[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c + 1e-30)   # parabolic vertex
            freqs.append(f[i] + d * binhz)
            mags.append(b - 0.25 * (a - c) * d)
    return np.array(freqs), np.array(mags)

def estimate_f0(y, fmin=40, fmax=2000):
    """Autocorrelation pitch estimate — the ear's temporal code, roughly."""
    y = y - y.mean()
    ac = fftconvolve(y, y[::-1])[len(y) - 1:]
    lo, hi = int(FS / fmax), int(FS / fmin)
    lag = lo + np.argmax(ac[lo:hi])
    if 1 <= lag < len(ac) - 1:
        a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
        lag = lag + 0.5 * (a - c) / (a - 2 * b + c + 1e-30)   # parabolic refine
    return FS / lag

def phase_delay(b, a, f0):
    """Phase delay (samples) of filter (b,a) at f0 — used to compensate the
    loop length so pitch stays on target when filters join the loop (§6.1/6.2)."""
    w = 2 * np.pi * f0 / FS
    _, h = freqz(b, a, worN=[w])
    return float(-np.angle(h[0]) / w)

def harmonic_table(y, f0_guess, nmax=12):
    """Measured partials as {n: (f_n, ratio f_n/(n·f1), level_dB)} — the core
    'do the modes land where theory says' proof used across S1–S4."""
    f_meas = estimate_f0(y)
    pf, pm = partials(y, f0_guess, fmax=(nmax + 1) * f0_guess)
    out = {}
    for fn, mn in zip(pf, pm):
        n = int(round(fn / f_meas))
        if 1 <= n <= nmax and n not in out:
            out[n] = (fn, fn / (n * f_meas), mn)
    return f_meas, out

def fit_inharmonicity(y, f0_guess, nmax=16, Bmax=3e-3):
    """Least-squares fit of B in f_n = n·f1·√(1+Bn²)/√(1+B). Returns (f1, B,
    ns, fns) — the S3 stiffness/dispersion proof."""
    pf, _ = partials(y, f0_guess, fmax=(nmax + 2) * f0_guess)
    if not len(pf):
        return f0_guess, 0.0, np.array([]), np.array([])
    f1 = pf[np.argmin(np.abs(pf - f0_guess))]
    ns, fns = [], []
    for fn in pf:
        n = int(round(fn / f1))
        if 1 <= n <= nmax and n not in ns:
            ns.append(n); fns.append(fn)
    ns, fns = np.array(ns), np.array(fns)
    Bs = np.linspace(0, Bmax, 3001)
    err = [np.sum((fns - ns * f1 * np.sqrt(1 + B * ns**2) / np.sqrt(1 + B))**2)
           for B in Bs]
    return f1, Bs[int(np.argmin(err))], ns, fns

def envelope_db(sig, win=2048):
    """Smoothed RMS envelope in dB — for reading decay slopes (S6)."""
    return 20 * np.log10(np.sqrt(
        np.convolve(sig**2, np.ones(win) / win, 'same')) + 1e-12)

def band_decay_db(y, lo_hz=900, hi_hz=2500, seg_s=0.25):
    """Energy drop (dB) in a low band vs a high band over the signal — the S2
    loss-as-filter proof (high band must die faster)."""
    b_lo, a_lo = butter(2, lo_hz / (FS / 2))
    b_hi, a_hi = butter(2, hi_hz / (FS / 2), btype='high')
    lo = lfilter(b_lo, a_lo, y); hi = lfilter(b_hi, a_hi, y)
    seg = int(seg_s * FS)
    def drop(sig):
        e0 = np.sqrt(np.mean(sig[:seg]**2)); e1 = np.sqrt(np.mean(sig[-seg:]**2))
        return 20 * np.log10(e1 / (e0 + 1e-30) + 1e-30)
    return drop(lo), drop(hi)

def centroid(y, fmax=8000):
    """Spectral centroid (Hz) — a rough brightness proxy (S7)."""
    f, db = spectrum_db(y)
    mag = 10**(db / 20)
    sel = f < fmax
    return np.sum(f[sel] * mag[sel]) / np.sum(mag[sel])
