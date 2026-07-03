"""
brass.py  --  Physical-modelling brass synthesis, first principles.

Companion code to BRASS_TRACK.md. Two layers, kept deliberately separate:

  ANALYSIS layer  (why brass instruments work)
      - transfer-matrix input impedance  -> where the resonances sit
      - Kelly-Lochbaum waveguide bore     -> same physics, time domain
    These PROVE the central claim of the charter: a bare cylinder gives an
    odd-only, non-harmonic resonance comb (useless); a flaring bell drags the
    low modes up into an (almost) full harmonic series with a weak/virtual
    fundamental. Both methods are validated against closed-form theory.

  SYNTHESIS layer (making a playable sound)
      - a self-oscillating waveguide voice (McIntyre-Schumacher-Woodhouse reed,
        the robust STK-Clarinet reflection form) primed into a chosen register
      - nonlinear wave-steepening for 'brassiness'
      - continuous fractional-delay slide (trombone)

    The self-oscillating LIP model (single-mass Bernoulli valve) is the
    physically 'correct' source, but a one-mass lip is a research-grade tuning
    problem to keep locked across the whole register (it drifts into its own
    high-frequency limit cycle). The reed-reflection source below oscillates
    reliably and tracks pitch, so it is what actually renders audio here.
    That honesty is deliberate (working-agreement principle 4).
"""
import numpy as np
from scipy.signal import butter, sosfilt

# --- physical constants -----------------------------------------------------
C_SOUND = 343.0      # m/s
RHO     = 1.204      # kg/m^3
FS      = 44100.0    # Hz


# ===========================================================================
#  small stateful filters
# ===========================================================================
class OnePole:
    """One-pole low-pass. Frequency-dependent loss: highs die faster than lows
       every round trip, exactly as real wall/radiation losses behave (2.4)."""
    def __init__(self, g=0.05): self.g = g; self.y = 0.0
    def tick(self, x):
        self.y = (1 - self.g) * x + self.g * self.y
        return self.y

class OneZero:
    """One-zero low-pass loss (gentler slope, used in the reed loop)."""
    def __init__(self, g=0.6): self.g = g; self.p = 0.0
    def tick(self, x):
        y = self.g * x + (1 - self.g) * self.p; self.p = x; return y

class DCBlock:
    """Removes the DC that a rectifying nonlinearity pumps into the loop (6.4)."""
    def __init__(self, R=0.99): self.R = R; self.x1 = 0.0; self.y1 = 0.0
    def tick(self, x):
        y = x - self.x1 + self.R * self.y1; self.x1 = x; self.y1 = y; return y


class Delay:
    """Fractional delay line, linear interpolation. The fractional read is what
       lets a trombone slide move *continuously* between pitches (2.8, 6.1)."""
    def __init__(self, maxlen=8192):
        self.buf = np.zeros(maxlen); self.n = maxlen; self.wr = 0; self.delay = 100.0
    def set_delay(self, d): self.delay = max(1.0, min(self.n - 2, d))
    def last_out(self):
        rd = self.wr - self.delay
        i = int(np.floor(rd)); frac = rd - i
        a = self.buf[i % self.n]; b = self.buf[(i + 1) % self.n]
        return a + frac * (b - a)
    def tick(self, x):
        o = self.last_out(); self.buf[self.wr % self.n] = x; self.wr += 1; return o


# ===========================================================================
#  ANALYSIS 1 -- transfer-matrix input impedance
# ===========================================================================
# Each short cylinder is a two-port (ABCD) relating (pressure, volume-flow) at
# its ends. Any flare/cone is just many short cylinders in cascade (2.5). The
# reed/lip sits at a pressure antinode, so the notes it can play are the MAXIMA
# of |Z_in| seen from the mouthpiece.
def cyl_matrix(f, length, area):
    w = 2 * np.pi * f
    a = np.sqrt(area / np.pi)
    alpha = (1.0 / (C_SOUND * a)) * np.sqrt(w) * 3.0e-5 * np.sqrt(2)   # wall loss
    k = w / C_SOUND - 1j * alpha
    Z0 = RHO * C_SOUND / area
    kl = k * length
    return np.array([[np.cos(kl),      1j * Z0 * np.sin(kl)],
                     [1j * np.sin(kl) / Z0, np.cos(kl)]], dtype=complex)

def radiation_load(f, area_end):
    """Unflanged open pipe radiation impedance, low-ka approximation."""
    a = np.sqrt(area_end / np.pi)
    k = 2 * np.pi * f / C_SOUND
    Z0 = RHO * C_SOUND / area_end
    ka = k * a
    return Z0 * ((ka ** 2) / 4.0 + 1j * 0.6133 * ka)

def input_impedance(freqs, profile):
    """profile: list of (length_m, area_m2) slices, mouthpiece end first."""
    Zin = np.zeros(len(freqs), dtype=complex)
    for i, f in enumerate(freqs):
        M = np.eye(2, dtype=complex)
        for (L, S) in profile:
            M = M @ cyl_matrix(f, L, S)
        Zrad = radiation_load(f, profile[-1][1])
        A, B, Cc, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        Zin[i] = (A * Zrad + B) / (Cc * Zrad + D)
    return Zin

def discretize(x, radius):
    """(x[m], radius[m]) polyline -> short-cylinder (length, area) slices."""
    return [ (x[i+1]-x[i], np.pi*(0.5*(radius[i]+radius[i+1]))**2)
             for i in range(len(x)-1) ]


# ===========================================================================
#  ANALYSIS 2 -- Kelly-Lochbaum waveguide bore (time domain)
# ===========================================================================
class KLBore:
    """N one-sample sections over a cross-sectional-area array. Between section
       i and i+1 the pressure reflection is  r = (A_i - A_{i+1})/(A_i + A_{i+1}).
       A widening (a cone, the bell) gives r<0: a partial, inverting reflection
       at every step. Those cumulative negative reflections are exactly what
       redistributes the modes into a harmonic series (2.3-2.5, 6.5).
       Validated: a bare cylinder rings at c/4L with clean odd harmonics."""
    def __init__(self, areas, loss_g=0.03, bell_reflect=0.9):
        self.N = len(areas); self.A = np.asarray(areas, float)
        self.k = (self.A[:-1] - self.A[1:]) / (self.A[:-1] + self.A[1:])
        self.pr = np.zeros(self.N); self.pl = np.zeros(self.N)
        self.loss = OnePole(loss_g); self.br = bell_reflect
    def step(self, inject, r_mouth=1.0):
        k = self.k
        w = k * (self.pr[:-1] - self.pl[1:])          # one-multiply scattering
        pr = np.empty(self.N); pl = np.empty(self.N)
        pr[1:] = self.pr[:-1] + w                     # right-goers advance
        pl[:-1] = self.pl[1:] + w                     # left-goers advance
        pr[0] = inject + r_mouth * self.pl[0]         # closed-ish mouthpiece
        pl[-1] = -self.br * self.loss.tick(self.pr[-1])  # bell: low-pass, invert
        self.pr, self.pl = pr, pl
        radiated = (1 - self.br) * self.pr[-1]
        return radiated, self.pr[0] + self.pl[0]

def impulse_response(bore, seconds=0.6, r_mouth=1.0):
    n = int(FS * seconds); out = np.zeros(n)
    for i in range(n):
        out[i], _ = bore.step(1.0 if i == 0 else 0.0, r_mouth)
    return out

def resonances(sig, height=0.05, min_hz=30, n_peaks=10):
    from scipy.signal import find_peaks
    w = np.hanning(len(sig))
    sp = np.abs(np.fft.rfft(sig * w)); fr = np.fft.rfftfreq(len(sig), 1 / FS)
    pk, _ = find_peaks(sp / sp.max(), height=height,
                       distance=int(min_hz * len(sig) / FS))
    return fr[pk][:n_peaks], (sp / sp.max())[pk][:n_peaks]


# ===========================================================================
#  bore geometry builders
# ===========================================================================
def cyl_areas(length, radius):
    dx = C_SOUND / FS
    N = max(4, int(round(length / dx)))
    return np.full(N, np.pi * radius * radius)

def trumpet_polyline(gamma=0.75, body=0.85, Lbell=0.28,
                     r_throat=0.002, r_bore=0.0055, r_mouth=0.062):
    """Leadpipe taper -> cylindrical body -> Bessel-horn bell flare.
       gamma/body tuned so modes 3-6 align near-harmonically (validated)."""
    xs, rs = [], []
    n = 40
    for i in range(n):
        t = i / (n - 1); xs.append(0.25 * t); rs.append(r_throat + (r_bore - r_throat) * t)
    x0 = 0.25
    n = 120
    for i in range(1, n):
        t = i / (n - 1); xs.append(x0 + body * t); rs.append(r_bore)
    x0 += body
    n = 90
    for i in range(1, n):
        t = i / (n - 1)
        r = r_bore + (r_mouth - r_bore) * (t ** (1 / (1 - gamma)))
        xs.append(x0 + Lbell * t); rs.append(r)
    return np.array(xs), np.array(rs)

def polyline_to_areas(xs, rs):
    dx = C_SOUND / FS
    N = int(round(xs[-1] / dx)); g = np.linspace(0, xs[-1], N)
    return np.pi * np.interp(g, xs, rs) ** 2


# ===========================================================================
#  B3 -- the mouthpiece (Helmholtz resonator / "brass formant")
# ===========================================================================
# A brass mouthpiece is a wide cup narrowing sharply to a small throat: a
# volume (cup) feeding a short narrow neck (throat/backbore) -- a Helmholtz
# resonator, f = (c/2pi)*sqrt(A/(V*L_eff)). Its single broad resonance sits
# ABOVE the playing modes (~800-1200 Hz for a trumpet) and boosts the response
# there: this is the "brass formant" that gives the family its bright ring and
# makes the upper register speak. Default geometry below pops at ~930 Hz,
# validated by the matched pop-test (radiate the cup+throat into free air and
# read its peak) and cross-checked against the analytic Helmholtz formula.
def mouthpiece_polyline(r_cup=0.009, cup_depth=0.011, r_throat=0.0018, throat_len=0.011):
    xs = np.array([0.0, cup_depth, cup_depth, cup_depth + throat_len])
    rs = np.array([r_cup, r_cup, r_throat, r_throat])
    return xs, rs

def prepend_mouthpiece(xs, rs, **kw):
    """Splice a mouthpiece onto the front (mouthpiece end) of a bore polyline."""
    mxs, mrs = mouthpiece_polyline(**kw)
    return np.concatenate([mxs, mxs[-1] + xs]), np.concatenate([mrs, rs])

def mouthpiece_pop(r_cup=0.009, cup_depth=0.011, r_throat=0.0018, throat_len=0.011):
    """The 'pop test': the cup+throat's own resonance, radiating into free air.
       This is exactly what a player hears popping a palm over the mouthpiece."""
    prof = [(cup_depth, np.pi * r_cup ** 2), (throat_len, np.pi * r_throat ** 2)]
    fr = np.linspace(50, 3000, 12000)
    Z = np.zeros(len(fr), dtype=complex)
    for i, f in enumerate(fr):
        M = np.eye(2, dtype=complex)
        for (L, S) in prof:
            M = M @ cyl_matrix(f, L, S)
        Zl = radiation_load(f, np.pi * r_throat ** 2)
        Z[i] = (M[0, 0] * Zl + M[0, 1]) / (M[1, 0] * Zl + M[1, 1])
    return fr[np.argmax(np.abs(Z))]


# ===========================================================================
#  SYNTHESIS -- self-oscillating waveguide voice
# ===========================================================================
def reed_table(x, offset=0.7, slope=-0.44):
    """Nonlinear valve reflection coefficient vs pressure difference. This is
       the reed/lip that COUPLES to the acoustic state of the bore -- the exact
       reason a linear/modal source cannot replace the waveguide (charter 2.1,
       working-agreement note). Clipped at 1.0 (valve fully open)."""
    return min(1.0, offset + slope * x)

class Voice:
    """Self-oscillating pressure-controlled valve on a bore. Primed into the
       target register (= 'setting the embouchure') so it locks to the intended
       mode instead of overblowing. Renders robust, pitch-tracking tones."""
    def __init__(self, freq, loss_g=0.6, refl=0.95, breath=0.5,
                 offset=0.7, slope=-0.44, prime=True, seed=1):
        self.d = Delay(); self.d.set_delay(FS / (2 * freq))
        self.lp = OneZero(loss_g); self.dc = DCBlock()
        self.refl = refl; self.breath = breath
        self.offset = offset; self.slope = slope
        self.rng = np.random.default_rng(seed)
        if prime:
            L = self.d.n; idx = np.arange(L)
            self.d.buf[:] = 0.05 * np.sin(2 * np.pi * freq * idx / FS)
            self.d.wr = int(np.ceil(self.d.delay)) + 1
    def set_delay_samples(self, d): self.d.set_delay(d)      # for the slide
    def tick(self, breath_env=1.0, noise=0.012):
        b = self.breath * breath_env
        b = b + b * noise * self.rng.standard_normal()
        out = -self.refl * self.lp.tick(self.d.last_out())   # reflect + loss
        pd = out - b                                         # pressure across reed
        rc = reed_table(pd, self.offset, self.slope)
        val = b + rc * pd                                    # MSW update
        self.d.tick(val)
        return self.dc.tick(val)

def render_note(freq, seconds=1.0, attack=0.05, release=0.1, **kw):
    v = Voice(freq, **kw)
    n = int(FS * seconds); out = np.zeros(n)
    a = int(FS * attack); r = int(FS * release)
    for i in range(n):
        env = min(1.0, i / max(1, a))
        if i > n - r: env *= max(0.0, (n - i) / r)
        out[i] = v.tick(env)
    return out


# ===========================================================================
#  SYNTHESIS -- brassiness (nonlinear wave steepening, 2.7)
# ===========================================================================
def brassiness(sig, drive=0.0):
    """A LINEAR waveguide can never sound brassy: it cannot create the high
       harmonics that a steepening (near-shock) wavefront produces at high blow.
       Real brass steepens because wave speed rises with local amplitude, so
       peaks catch up on troughs. This is the reduced 1-D model of that effect:
       an amplitude-dependent waveshaper whose harmonic generation grows with
       playing level. drive=0 -> transparent (pp); large drive -> bright (ff).
       (Full physics = a Burgers/Menguy-Gilbert nonlinear propagation stage.)"""
    if drive <= 0: return sig
    x = sig / (np.max(np.abs(sig)) + 1e-9)
    y = np.tanh(x * (1 + drive)) / np.tanh(1 + drive)   # level-dependent steepening
    return y * (np.max(np.abs(sig)) + 1e-9)


# ===========================================================================
#  SYNTHESIS -- trombone slide (continuous fractional delay, 2.8)
# ===========================================================================
def render_slide(f_start, f_end, seconds=1.2, smooth_ms=8.0, **kw):
    """A slide is not a series of discrete pitches -- it is one bore whose
       LENGTH changes continuously. Only a fractional delay can do that; the
       delay length is smoothed to avoid zipper noise on the moving read."""
    v = Voice(f_start, **kw)
    n = int(FS * seconds); out = np.zeros(n)
    d0 = FS / (2 * f_start); d1 = FS / (2 * f_end)
    a = 1.0 - np.exp(-1.0 / (smooth_ms * 0.001 * FS))    # one-pole smoother
    d = d0
    for i in range(n):
        t = i / n
        target = FS / (2 * (f_start + (f_end - f_start) * t))
        d += a * (target - d); v.set_delay_samples(d)
        env = min(1.0, i / (0.04 * FS))
        out[i] = v.tick(env)
    return out


# ===========================================================================
#  B7 -- mutes + directivity (2.9)
# ===========================================================================
# MUTE. An object wedged in the bell does two audible things: (1) it blocks the
# wide-band low-frequency escape, so lows reflect back down the bore instead of
# radiating -> a high-passed, pinched, nasal timbre; (2) it adds its OWN cavity
# resonance, a strong mid/high formant that is the mute's characteristic
# "voice". The felt back-reaction slightly retunes the horn (players lip it
# back), but the audible signature is this radiation-side colouring, modelled
# here on the radiated output. Each mute is a (formant peak, sharpness Q,
# low-cut, boost) fingerprint.
MUTES = {
    "straight": dict(peak=1800.0, Q=2.5, hp=600.0, gain=1.6),  # bright, pinched
    "cup":      dict(peak=1100.0, Q=1.4, hp=350.0, gain=1.0),  # darker, veiled
    "harmon":   dict(peak=2400.0, Q=4.0, hp=900.0, gain=2.2),  # hollow, very bright
}

def apply_mute(sig, kind="straight"):
    p = MUTES[kind]
    sos_hp = butter(2, p["hp"] / (FS / 2), btype="high", output="sos")
    y = sosfilt(sos_hp, sig)                                   # kill low escape
    bw = p["peak"] / p["Q"]
    lo = max(50.0, p["peak"] - bw / 2) / (FS / 2)
    hi = min(FS / 2 - 1, p["peak"] + bw / 2) / (FS / 2)
    sos_bp = butter(2, [lo, hi], btype="band", output="sos")
    return y + p["gain"] * sosfilt(sos_bp, sig)                # + mute's formant

# DIRECTIVITY. The bell is a finite radiator: long wavelengths (low f) diffract
# in every direction, short wavelengths (high f) beam forward along the axis.
# So an on-axis listener hears a brighter tone than an off-axis one -- why a
# trumpet aimed at you is piercing and dull from the side. Modelled as a
# forward-gain low-pass whose cutoff falls as you move off axis.
def directivity(sig, angle_deg=0.0):
    fc = 8000.0 * np.cos(np.radians(angle_deg)) ** 2 + 400.0
    sos = butter(2, min(FS / 2 - 1, fc) / (FS / 2), btype="low", output="sos")
    return sosfilt(sos, sig)


if __name__ == "__main__":
    # smoke test
    cyl = KLBore(cyl_areas(0.5, 0.006), loss_g=0.05, bell_reflect=0.999)
    f, _ = resonances(impulse_response(cyl))
    print("bare cylinder resonances:", np.round(f[:6], 1), " (odd harmonics)")
    tone = render_note(220.0, 0.4)
    print("220 Hz tone rms:", round(float(np.sqrt(np.mean(tone**2))), 3))
    print("mouthpiece pop:", round(mouthpiece_pop(), 0), "Hz  (brass formant)")
    print("mute (straight) rms:", round(float(np.sqrt(np.mean(apply_mute(tone, 'straight')**2))), 3))
