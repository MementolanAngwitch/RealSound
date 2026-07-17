"""
realsound.py -- validated physics for the RealSound synth (backend module).

Import into the notebook to use/test:  `from realsound import *`

Only STABLE, verified functions live here. New/experimental physics stays in the
notebook as functions until it settles, then graduates into this file.

Design rule: every function is pure, with all inputs as explicit parameters (no
notebook globals like fs/t/f_0/idx). Clean imports, and a near-mechanical port to
C++/Rust later.
"""

import numpy as np
import scipy.linalg
import scipy.signal


# ============================================================ STRING ==========

def pluck_string(f, beta=0.5, duration=2.0, fs=44100, rho=1.0):
    """Karplus-Strong plucked string with fractional-delay tuning.

    f=pitch(Hz), beta=pluck position(0-1), rho=loop gain/sustain(<1).
    Verified <0.5 cents. The fractional interpolation lives INSIDE the feedback
    loop (buf[w]), which is what actually retunes the string.
    """
    D = fs / f - 0.5                      # loop delay to realize (0.5 = avg filter)
    N = int(D)
    frac = D - N
    p = int(beta * N)
    pluck = np.concatenate((np.linspace(0, 1, p), np.linspace(1, 0, N - p)))
    L = N + 2
    buf = np.zeros(L); buf[:N] = pluck
    out = np.zeros(int(fs * duration))
    w = 0; last = 0.0
    for i in range(len(out)):
        r0 = (w - N) % L                 # tap at delay N
        r1 = (w - N - 1) % L             # tap at delay N+1 (older tap gets frac)
        y = (1 - frac) * buf[r0] + frac * buf[r1]
        buf[w] = 0.5 * (y + last) * rho  # interpolated y feeds the loop -> retunes
        last = y; out[i] = y; w = (w + 1) % L
    return out


# =================================================== PLATE (offline solvers) ==

def orthotropic_plate(ax=4, ay=1, resolution=20):
    """Square orthotropic plate: K = L_a @ L_a, simply-supported.
    ax/ay = directional stiffness (grain). Returns (K, evals, evecs)."""
    Nx = Ny = resolution
    N = Nx * Ny
    def idx(i, j): return i * Ny + j
    La = np.zeros((N, N))
    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j)
            La[p, p] = -(2 * ax + 2 * ay)
            if i > 0:      La[p, idx(i - 1, j)] = ax
            if i < Nx - 1: La[p, idx(i + 1, j)] = ax
            if j > 0:      La[p, idx(i, j - 1)] = ay
            if j < Ny - 1: La[p, idx(i, j + 1)] = ay
    K = La @ La
    evals, evecs = scipy.linalg.eigh(K)
    return K, evals, evecs


def boolean_8_grid(res=80, cxTop=0.5, cyTop=0.68, rTop=0.18,
                   cxBot=0.5, cyBot=0.35, rBot=0.25,
                   xHole=0.5, yHole=0.52, rHole=0.05):
    """Figure-8 guitar-body outline with soundhole. Returns bool (res, res)."""
    x = np.linspace(0, 1, res); y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    top  = (X - cxTop)**2 + (Y - cyTop)**2 <= rTop**2
    bot  = (X - cxBot)**2 + (Y - cyBot)**2 <= rBot**2
    hole = (X - xHole)**2 + (Y - yHole)**2 <= rHole**2
    return (top | bot) & ~hole


def plate_from_mask(mask, ax=4, ay=1):
    """Plate over an arbitrary boolean mask (any body shape).
    Returns (evals, evecs, idx) where idx is the grid->matrix-row index map
    (-1 = not a DOF). Carry idx to plot modes or look up strike/listen points."""
    interior = np.argwhere(mask)
    idx = -np.ones(mask.shape, dtype=int)
    for row, (i, j) in enumerate(interior):
        idx[i, j] = row
    M = len(interior)
    L = np.zeros((M, M))
    for row, (i, j) in enumerate(interior):
        L[row, row] = -(2 * ax + 2 * ay)
        for di, dj, w in [(-1, 0, ax), (1, 0, ax), (0, -1, ay), (0, 1, ay)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj]:
                L[row, idx[ni, nj]] = w
    K = L @ L
    evals, evecs = scipy.linalg.eigh(K)
    return evals, evecs, idx


def three_oscillator_model(f_top=180, f_back=200, m_top=0.15, m_back=0.05,
                           Volume=0.012, hole_radius=0.041, hole_thickness=0.003,
                           top_area=0.18, back_area=0.013, rho=1.2, c=343):
    """Coupled top-plate / back-plate / soundhole-air cavity.
    Cavity spring = (rho c^2 / V) * a a^T with a = [A_top, A_back, -A_hole].
    Returns (freqs_Hz, evecs); evecs[:,k] = [top, back, air] participation."""
    omega_top  = 2 * np.pi * f_top
    omega_back = 2 * np.pi * f_back
    hole_area  = np.pi * hole_radius**2
    L_eff      = hole_thickness + 1.2 * hole_radius
    m_hole     = rho * hole_area * L_eff
    a = np.array([top_area, back_area, -hole_area])
    M = np.diag([m_top, m_back, m_hole])
    K = np.diag([m_top * omega_top**2, m_back * omega_back**2, 0.0]) \
        + (rho * c**2 / Volume) * np.outer(a, a)
    evals, evecs = scipy.linalg.eigh(K, M)
    return np.sqrt(evals) / (2 * np.pi), evecs


# =================================================== MODAL SYNTHESIS ==========

def modal_bank(freqs, amps, Qs, t):
    """Sum of decaying sinusoids. t = time vector (np.arange(int(fs*dur))/fs).
    tau = Q/(pi f): higher Q -> longer ring / narrower peak (bell vs thud)."""
    y = np.zeros_like(t)
    for f, a, Q in zip(freqs, amps, Qs):
        tau = Q / (np.pi * f)
        y += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    return y


def build_body_IR(evals, evecs, idx, t, n_modes=30, Q=20,
                  sx=None, sy=None, lx=None, ly=None,
                  f_top=180, g_low=0.1, **osc):
    """Full acoustic-body impulse response: the higher distributed plate modes
    plus the three coupled low modes (top/back/air) from three_oscillator_model.

    idx : the grid->matrix-row map from plate_from_mask (an ARRAY).
          For a plain square body, pass plate_from_mask(np.ones((res,res), bool)).
    sx,sy / lx,ly : strike / listen grid cells; default to interior points.
    g_low : balances low-end boom (coupled modes) vs plate coloration.
    **osc : forwarded to three_oscillator_model (Volume, hole_radius, ...).
    """
    inside = np.argwhere(idx >= 0)                  # cells inside the body
    if sx is None:
        sx, sy = inside[len(inside) // 3]           # off-center interior points
        lx, ly = inside[2 * len(inside) // 3]
    r_s, r_l = idx[sx, sy], idx[lx, ly]             # array indexing -> matrix rows

    # higher plate modes (skip mode 0 -- fundamental is handled by the coupling)
    plate_f = f_top * np.sqrt(evals) / np.sqrt(evals[0])
    freqs_p, amps_p, Qs_p = [], [], []
    for k in range(1, n_modes):
        freqs_p.append(plate_f[k])
        amps_p.append(evecs[r_s, k] * evecs[r_l, k])
        Qs_p.append(Q)

    # three coupled low modes: drive = top component, radiate = top + air
    freqs3, ev3 = three_oscillator_model(f_top=f_top, **osc)
    amps3 = [ev3[0, k] * (ev3[0, k] + ev3[2, k]) for k in range(3)]
    Qs3 = [30, 30, 15]

    freqs = np.concatenate([freqs3, freqs_p])
    amps  = np.concatenate([g_low * np.array(amps3), amps_p])
    Qs    = np.concatenate([Qs3, Qs_p])
    y = modal_bank(freqs, amps, Qs, t)
    return y / np.max(np.abs(y))


# ======================================================= SEQUENCING ==========

def make_scale(root_hz, semitones):
    """Equal temperament: f = root * 2^(k/12). semitones = list of offsets."""
    return root_hz * 2 ** (np.array(semitones) / 12)


def play_sequence(freqs, voice, dt=0.4, fs=44100):
    """Overlap-add notes onto a timeline. voice: f -> samples (any instrument)."""
    notes = [voice(f) for f in freqs]
    onsets = [round(i * dt * fs) for i in range(len(freqs))]
    total = max(o + len(n) for o, n in zip(onsets, notes))
    out = np.zeros(total)
    for o, n in zip(onsets, notes):
        out[o:o + len(n)] += n
    return out / np.max(np.abs(out))


# ========================================================== ANALYSIS =========

def measure_frequency(sig, fs=44100, skip=2000):
    """Fundamental (Hz) of rendered audio, sub-sample accurate.
    DC-remove -> autocorrelation -> parabolic peak refine (cents-level checks)."""
    s = sig[skip:]
    s = s - np.mean(s)
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    lo = 30
    k = np.argmax(ac[lo:]) + lo
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c)
    return fs / (k + d)
