"""
string_synth.py — the SYNTHESIS engine for the string track.

The other half of the brass-style split: this is *making it playable*. It holds
the sound-producing models — plucked (pluck-position waveguide + single-loop
Karplus–Strong), bowed (MSW stick–slip), hammered (nonlinear felt), and coupled
strings (polarizations / unisons) — plus the synthetic body IR for commuted
synthesis. DSP nodes and the phase-delay tuning helper are imported from the
analysis engine so the two stay in exact sync.

Like brass.py, this library is pure signal-processing math: it returns numpy
arrays and NEVER touches a sound card or writes a file. Rendering to WAV lives
in render_assets.py; inline playback lives in the notebook (§5).
"""
import numpy as np
from scipy.signal import butter, fftconvolve, lfilter as _lf

from string_analysis import (FS, DelayLine, OnePole, AllpassChain, FracDelay,
                             loss_filter, phase_delay)

# ==========================================================================
# S0 — Karplus–Strong plucked-string baseline
# ==========================================================================
def karplus_strong(f0, dur_s=1.5, loss=0.998):
    """Minimal waveguide = the plucked string (§6.5). Noise burst = broadband
    pluck initial condition; 2-point averager = crude frequency-dependent loss.
    Returns a numpy array."""
    N = int(round(FS / f0))
    dur = int(dur_s * FS)
    buf = np.random.uniform(-1, 1, N)          # pluck: energy at ALL frequencies
    y = np.zeros(dur)
    for t in range(dur):
        y[t] = buf[t % N]
        buf[t % N] = 0.5 * (buf[t % N] + buf[(t + 1) % N]) * loss
    return y

# ==========================================================================
# S1–S3 — explicit two-delay-line string (pluck position, loss, dispersion)
# ==========================================================================
def pluck_string(f0, pluck_frac=0.20, pick_frac=0.85, dur_s=1.5,
                 loss_scalar=None, fc=None, gain=0.999, allpass=None):
    """Two travelling-wave delay lines, fixed–fixed (r = −1 both ends).

    Excitation is an INITIAL CONDITION: a triangular displacement plucked at
    `pluck_frac`·L, split 50/50 onto the two waves (zero initial velocity).
    Harmonic amps ∝ sin(nπβ)/n² → a comb that nulls harmonics with a node at
    the pluck point (§2.1–2.3).

      loss_scalar : S1 placeholder scalar loss (set this OR fc)
      fc, gain    : S2 frequency-dependent loss LP (set fc to enable)
      allpass     : (a, M) tuple → S3 dispersion all-pass chain in the loop
    """
    lp = ap = None
    tau = 0.0
    if fc is not None:
        lp, b_lp, a_lp = loss_filter(fc, gain)
        tau += phase_delay(b_lp, a_lp, f0)
    if allpass is not None:
        ap = AllpassChain(*allpass)
        tau += ap.phase_delay(f0)
    budget = FS / f0 - tau
    N = int(budget // 2)
    fd = FracDelay(budget - 2 * N)                 # fractional loop tuning (§6.1)
    p = int(pluck_frac * N); pick = int(pick_frac * N)
    g = loss_scalar if loss_scalar is not None else 1.0

    x = np.arange(N, dtype=float)
    tri = np.where(x <= p, x / p, (N - 1 - x) / (N - 1 - p))
    r_wave, l_wave = 0.5 * tri.copy(), 0.5 * tri.copy()

    dur = int(dur_s * FS)
    y = np.zeros(dur)
    for t in range(dur):
        out_b = r_wave[-1]; out_n = l_wave[0]
        y[t] = r_wave[pick] + l_wave[pick]         # observable = superposition
        r_wave[1:] = r_wave[:-1]; r_wave[0] = -out_n   # nut: fixed end, r = −1
        l_wave[:-1] = l_wave[1:]
        back = fd.tick(out_b)                       # bridge return: r = −1 + …
        if ap is not None:
            back = ap.tick(back)                   # … dispersion all-pass (S3)
        if lp is not None:
            back = lp.tick(back)                   # … frequency-dependent loss (S2)
        l_wave[-1] = -g * back
    return y

# ==========================================================================
# S4 — bowed string (MSW stick–slip friction ↔ string feedback)
# ==========================================================================
def _solve_slip(dh, Z, Fb, mus, mud, vc, prev):
    """Intersect friction curve with string load line. Returns (F, Δv).
    STICK if the force to hold v = v_bow is within the static limit; else the
    load line into the friction curve gives a quadratic in Δv, solved closed
    form. The ambiguous double-crossing region is resolved by continuity with
    the previous sample (crude hysteresis — MSW discuss this)."""
    F_stick = -2 * Z * dh
    if abs(F_stick) <= mus * Fb:
        return F_stick, 0.0                        # STICK: string rides the bow
    s = np.sign(F_stick); dh_m = s * dh            # mirror so slip branch Δv < 0
    a2 = 2 * Z
    b2 = -(2 * Z * (vc + dh_m) + Fb * mud)
    c2 = vc * (2 * Z * dh_m + Fb * mus)
    disc = b2 * b2 - 4 * a2 * c2
    if disc < 0:
        F = mus * Fb
        return s * F, s * (dh_m + F / (2 * Z))
    rt = np.sqrt(disc)
    roots = [r for r in ((-b2 - rt) / (2 * a2), (-b2 + rt) / (2 * a2))
             if r <= 1e-12]
    if not roots:
        F = mus * Fb
        return s * F, s * (dh_m + F / (2 * Z))
    d = min(roots, key=lambda r: abs(r - s * prev))
    return s * 2 * Z * (d - dh_m), s * d

def bow_string(f0=196.0, beta=1/8, F_b=1.20, v_bow=0.20,
               mus=0.8, mud=0.3, vc=0.05, fc=6500, gain=0.995, dur_s=2.0,
               return_stick=False):
    """Self-oscillating bowed string. The bow at position β splits the string
    into two segments, each collapsing to one loop (r = −1 far end). Helmholtz
    motion (one stick + one slip per period, sawtooth at the bridge) EMERGES
    from the friction↔string feedback; it is not programmed (§2.6).

    F_b is the 'bow pressure' knob: too small → multiple slips/period ('surface
    sound'); enough → locked Helmholtz. Returns the bridge velocity wave (and
    the per-sample stick mask if return_stick)."""
    Ltot = FS / f0
    half = int(Ltot // 2)
    N2 = max(4, int(round(beta * half))); N1 = half - N2
    lp, b_lp, a_lp = loss_filter(fc, gain)
    fd = FracDelay(max(0.0, Ltot - 2 * half - phase_delay(b_lp, a_lp, f0)))
    left = DelayLine(2 * N1 + 4); right = DelayLine(2 * N2 + 4)
    Z0 = 1.0
    dur = int(dur_s * FS)
    y = np.zeros(dur); stick = np.zeros(dur, bool); prev_d = 0.0
    for t in range(dur):
        v_in_l = -left.read(2 * N1)                # nut: single r = −1
        v_in_r = -lp.tick(fd.tick(right.read(2 * N2)))   # bridge: r = −1 + loss
        F, prev_d = _solve_slip(v_in_l + v_in_r - v_bow, Z0, F_b,
                                mus, mud, vc, prev_d)
        stick[t] = (prev_d == 0.0)
        inj = F / (2 * Z0)                         # force → velocity, both ways
        left.write(v_in_r + inj)
        right.write(v_in_l + inj)
        y[t] = right.read(N2)                      # velocity wave at the bridge
    return (y, stick) if return_stick else y

# ==========================================================================
# S5 — single-loop string + synthetic body (commuted synthesis)
# ==========================================================================
def string_loop(exc, f0, fc=5500, gain=0.999, dur_s=2.0):
    """Single-loop string (both r = −1 folded to net +1), driven by an
    arbitrary excitation. Used for commuted synthesis: fold the body IR into
    the excitation, then run the cheap loop (§ commuted synthesis)."""
    lp, b_, a_ = loss_filter(fc, gain)
    Ldl = FS / f0 - phase_delay(b_, a_, f0)
    Ni = int(Ldl)
    dl = DelayLine(Ni + 4)
    fd = FracDelay(Ldl - Ni)
    n = int(dur_s * FS)
    xin = np.zeros(n); xin[:min(len(exc), n)] = exc[:min(len(exc), n)]
    y = np.zeros(n)
    for t in range(n):
        v = xin[t] + lp.tick(fd.tick(dl.read(Ni)))
        dl.write(v)
        y[t] = v
    return y

def make_body_ir():
    """Synthetic guitar-ish body impulse response: damped modes (freq Hz,
    relative amp, decay ms) + a little attack noise. The body's IRREGULAR
    resonances are the instrument's identity. HONEST LIMITATION: synthetic —
    swap in a MEASURED body IR for instant realism."""
    modes = [(98, 1.0, 90), (192, 0.9, 70), (245, 0.45, 60), (400, 0.4, 45),
             (530, 0.35, 40), (720, 0.30, 30), (1010, 0.22, 25),
             (1400, 0.15, 20)]
    t_ir = np.arange(int(0.20 * FS)) / FS
    ir = sum(a * np.exp(-t_ir / (tau / 1000)) * np.sin(2 * np.pi * f * t_ir)
             for f, a, tau in modes)
    ir += 0.02 * np.exp(-t_ir / 0.004) * np.random.randn(len(t_ir))
    return ir / np.max(np.abs(ir))

def pluck_excitation(n_noise=40, fc=4000, length_s=0.03):
    """Short broadband pluck excitation (pre-body), for commuted synthesis."""
    raw = np.zeros(int(length_s * FS)); raw[:n_noise] = np.random.uniform(-1, 1, n_noise)
    return _lf(*butter(1, fc / (FS / 2)), raw)

# ==========================================================================
# S6 — coupled strings (two polarizations / unison pair)
# ==========================================================================
def two_polarizations(f0, dur_s=3.0):
    """One string, two transverse planes: a fast bridge-loaded vertical
    polarization + a slow, barely-coupled, slightly-detuned horizontal one.
    Their sum has the characteristic TWO-STAGE DECAY (fast attack-decay, then
    a long quiet 'aftersound')."""
    vert = string_loop(_lf(*butter(1, 3500 / (FS / 2)),
                           np.r_[np.random.uniform(-1, 1, 50), np.zeros(10)]),
                       f0, fc=5000, gain=0.992, dur_s=dur_s)
    horiz = string_loop(np.r_[np.random.uniform(-1, 1, 50), np.zeros(10)],
                        f0 * 1.0007, fc=3500, gain=0.9997, dur_s=dur_s)
    return 0.85 * vert + 0.25 * horiz

def coupled_pair(f0a, f0b, ga, gb, eps, dur_s=3.0, fc=5000):
    """Two detuned unison strings exchanging a little energy each sample through
    a shared bridge (ε cross-feed) → beating + shimmer. HONEST LIMITATION: real
    coupling is via the bridge's shared mechanical impedance; ε is a
    phenomenological stand-in for that behaviour."""
    lpa, ba, aa = loss_filter(fc, ga); lpb, bb, ab = loss_filter(fc, gb)
    La = FS / f0a - phase_delay(ba, aa, f0a)
    Lb = FS / f0b - phase_delay(bb, ab, f0b)
    Na, Nb = int(La), int(Lb)
    da, db = DelayLine(Na + 4), DelayLine(Nb + 4)
    fda, fdb = FracDelay(La - Na), FracDelay(Lb - Nb)
    n = int(dur_s * FS)
    exc = np.zeros(n)
    exc[:50] = _lf(*butter(1, 3500 / (FS / 2)), np.random.uniform(-1, 1, 50))
    y = np.zeros(n)
    for t in range(n):
        fa = lpa.tick(fda.tick(da.read(Na)))
        fb = lpb.tick(fdb.tick(db.read(Nb)))
        va = exc[t] + (1 - eps) * fa + eps * fb
        vb = exc[t] + (1 - eps) * fb + eps * fa
        da.write(va); db.write(vb)
        y[t] = va + vb
    return y

# ==========================================================================
# S7 — piano hammer (transient nonlinear felt contact)
# ==========================================================================
def hammer_strike(v0, f0=110.0, beta=1/8, K=6.0, pexp=2.5, m=60.0, dur_s=2.0):
    """Felt hammer = a stiffening spring F = K·cᵖ (p > 1) on a free-flying mass,
    striking the string at β. Velocity → brightness EMERGES: harder strike →
    deeper compression → stiffer spring → shorter contact → wider spectrum.
    HONEST LIMITATION: elastic felt (real felt is hysteretic — Stulov model).
    Returns (array, contact_samples)."""
    half = int(FS / f0 // 2)
    N2 = max(4, int(round(beta * half))); N1 = half - N2
    lp, b_, a_ = loss_filter(6000, 0.996)
    fd = FracDelay(max(0.0, FS / f0 - 2 * half - phase_delay(b_, a_, f0)))
    left, right = DelayLine(2 * N1 + 4), DelayLine(2 * N2 + 4)
    y_h, v_h_ham = -1e-6, v0                       # hammer just below, moving up
    y_s = 0.0                                       # string displ. at strike point
    n = int(dur_s * FS)
    y = np.zeros(n); contact = 0
    for t in range(n):
        v_in_l = -left.read(2 * N1)
        v_in_r = -lp.tick(fd.tick(right.read(2 * N2)))
        vh = v_in_l + v_in_r
        c = y_h - y_s
        F = K * c**pexp if c > 0 else 0.0          # felt: stiffening spring
        if c > 0:
            contact += 1
        inj = F / 2.0                              # Z0 = 1
        v = vh + inj
        y_s += v                                    # ∫v dt (dt = 1 sample)
        v_h_ham -= F / m                            # Newton on the hammer
        y_h += v_h_ham
        left.write(v_in_r + inj)
        right.write(v_in_l + inj)
        y[t] = right.read(N2)
    return y, contact
