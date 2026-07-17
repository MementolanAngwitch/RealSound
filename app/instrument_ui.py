
import functools
import os
import re
import tempfile
import zipfile
import numpy as np
import scipy.linalg
import scipy.signal
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")            # headless: we render figures to images, never a window
import matplotlib.pyplot as plt

fs = 44100                       # sample rate (Hz)


def safe_name(name, fallback="realsound"):
    """Sanitise a user-typed filename into a safe basename (no path escapes)."""
    cleaned = re.sub(r"[^\w.-]", "_", (name or "").strip())
    return cleaned or fallback


# ----------------------------------------------------------------------------
# 1. THE STRING  (Karplus-Strong / digital waveguide)
# ----------------------------------------------------------------------------
# A plucked string is a delay line of length N fed back through a 2-point
# averaging low-pass. The loop length sets the pitch: one trip round the loop
# is one period, so f ~= fs / (N + 0.5). The 0.5 is the half-sample group
# delay the averaging filter adds. beta is WHERE you pluck (fraction along the
# string); it sets the triangular initial shape, which controls which
# harmonics are strong. rho (<1) bleeds energy out of the loop each pass, so
# the note decays -- that is the string's damping.

def note_to_N(freq_hz, fs=fs):
    """Integer delay length for a pitch, only used to size the render buffer.

    The *audible* pitch is now set exactly by the fractional delay in
    pluck_string; this integer N is just a convenient handle for the T60/length
    calculation in sustain_duration."""
    return max(2, int(round(fs / freq_hz - 0.5)))


def pluck_string(beta=0.5, f=220.0, duration=2.0, fs=fs, rho=1.0):
    """Fractional-delay Karplus-Strong: pitch is set EXACTLY, not quantised.

    Physics: pitch needs a loop delay of D = fs/f - 0.5 samples, but a plain
    ring buffer can only delay by a whole number of samples, so the old
    round(D) shaved the tuning (worse up high, where one sample is a big slice
    of the period). Here we split D into an integer part N = floor(D) and a
    fraction frac, and read the delay line by LINEARLY INTERPOLATING between the
    tap at N and the tap at N+1:  y = (1-frac)*d[N] + frac*d[N+1]. That
    fractional read places the loop delay exactly at D, so every note is in
    tune. The 0.5*(y+last)*rho feedback is the same 2-point averaging low-pass
    and loss as before."""
    D = fs / f - 0.5
    N = int(D)
    frac = D - N

    p = min(max(int(beta * N), 1), N - 1)
    pluck = np.concatenate((np.linspace(0, 1, p), np.linspace(1, 0, N - p)))

    L = N + 2                                 # buffer a bit longer than the max tap
    buf = np.zeros(L)
    buf[:N] = pluck
    out = np.zeros(int(fs * duration))
    w = 0                                     # write index
    last = 0.0
    for i in range(len(out)):
        r0 = (w - N) % L                      # tap at delay N
        r1 = (w - N - 1) % L                  # tap one sample older (N+1)
        y = (1 - frac) * buf[r0] + frac * buf[r1]   # fractional (interpolated) read
        buf[w] = 0.5 * (y + last) * rho       # 2-point average + loss
        last = y
        out[i] = y
        w = (w + 1) % L
    return out


def sustain_duration(rho=0.99, N=100, fs=fs):
    """How long the string actually rings, from the loop's T60.

    Each loop pass multiplies amplitude by ~rho, so amplitude decays like
    rho^(t*fs/N). Solving for a 60 dB drop gives T60 = -6.91*N / (fs*ln rho).
    We render 1.2*T60 so the tail isn't chopped."""
    T60 = -6.91 * N / (fs * np.log(rho))
    return 1.2 * T60


# ----------------------------------------------------------------------------
# 2. THE BODY  (orthotropic plate, modal synthesis)
# ----------------------------------------------------------------------------
# The guitar top is a stiff plate. Its motion obeys a biharmonic (fourth-order)
# equation; discretised, the stiffness matrix K = L @ L where L is an
# anisotropic Laplacian. ax/ay are the bending stiffnesses along/across the
# grain -- wood is stiffer along the grain, which splits the degenerate mode
# pairs of an isotropic plate. The eigenvectors of K are the mode shapes; the
# eigenvalues give the mode frequencies. Those depend ONLY on (ax, ay,
# resolution), so we cache them: striking or listening somewhere else does not
# change the modes, only how much each one is excited/heard.

# --- Body outline (mask) generators -----------------------------------------
# A real top is not a square. We restrict the plate to a boolean mask of the
# body outline; the finite-difference operator is then built ONLY on the
# interior cells, with every neighbour that falls outside the mask treated as
# pinned (w = 0). Different masks = different bodies. (eight_mask/circle_mask
# are verbatim from the notebook; X is grid axis 0, Y is axis 1, indexing='ij'.)

def eight_mask(res=40, cxTop=0.5, cyTop=0.68, rTop=0.18, cxBot=0.5, cyBot=0.35,
               rBot=0.25, xHole=0.5, yHole=0.52, rHole=0.05):
    """Rounded figure-8 guitar top: a small upper bout ∪ a bigger lower bout,
    with the soundhole disc cut OUT of the plate mesh."""
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    topBody = (X - cxTop) ** 2 + (Y - cyTop) ** 2 <= rTop ** 2
    botBody = (X - cxBot) ** 2 + (Y - cyBot) ** 2 <= rBot ** 2
    soundhole = (X - xHole) ** 2 + (Y - yHole) ** 2 <= rHole ** 2
    body = topBody | botBody
    body = body & ~soundhole
    return body


def circle_mask(res=40, cx=0.5, cy=0.5, r=0.4):
    """A circular drum-head body, for comparison."""
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, y, indexing='ij')
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def plate_from_mask(mask, ax=4.0, ay=1.0):
    """Biharmonic plate on an arbitrary outline via an index map.

    `interior` lists the cells inside the mask; `idx` maps a grid cell (i,j) to
    its matrix row (or -1 outside). We assemble the anisotropic Laplacian L only
    over interior cells, connecting to a neighbour only when that neighbour is
    also inside the mask (otherwise the edge is dropped -> pinned boundary).
    Then K = L@L and we diagonalise. Returns (evals, evecs, idx) where idx is
    the 2-D index-map ARRAY (indexed with [], not called)."""
    interior = np.argwhere(mask)                 # matrix row -> grid cell
    idx = -np.ones(mask.shape, dtype=int)        # grid cell -> matrix row (-1 outside)
    for row, (i, j) in enumerate(interior):
        idx[i, j] = row
    M = len(interior)
    L = np.zeros((M, M))
    for row, (i, j) in enumerate(interior):
        L[row, row] = -(2 * ax + 2 * ay)         # diagonal of the anisotropic stencil
        for (di, dj, w) in [(-1, 0, ax), (1, 0, ax), (0, -1, ay), (0, 1, ay)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1] and mask[ni, nj]:
                L[row, idx[ni, nj]] = w           # connect only to interior neighbours
    K = L @ L
    evals, evecs = scipy.linalg.eigh(K)
    return evals, evecs, idx


MASKS = {
    "Guitar (figure-8)": eight_mask,
    "Circle": circle_mask,
    "Rectangle": lambda res: np.ones((res, res), dtype=bool),   # full grid = old square
}


@functools.lru_cache(maxsize=32)
def plate_modes(ax=4.0, ay=1.0, resolution=30, shape="Guitar (figure-8)"):
    """Cached body modes for a shape. Returns (evals, evecs, idx_map).
    Cached by (ax, ay, resolution, shape) — the modes depend only on these."""
    mask = MASKS.get(shape, eight_mask)(int(resolution))
    return plate_from_mask(mask, ax, ay)


def _frac_to_row(pt, idx_map):
    """Map a fractional plate coordinate (0..1, 0..1) to a matrix row.
    If the requested cell is outside the mask (or in the soundhole), snap to the
    nearest interior cell so strike/listen always land on the body."""
    res = idx_map.shape[0]
    i = min(max(int(round(pt[0] * (res - 1))), 0), res - 1)
    j = min(max(int(round(pt[1] * (res - 1))), 0), res - 1)
    if idx_map[i, j] >= 0:
        return int(idx_map[i, j])
    ii, jj = np.where(idx_map >= 0)              # nearest interior cell
    k = int(np.argmin((ii - i) ** 2 + (jj - j) ** 2))
    return int(idx_map[ii[k], jj[k]])


def plate_ir(evals, evecs, idx_map, f0=220.0, Q=20.0,
             strike=(0.5, 0.5), listen=(0.5, 0.5),
             n_modes=10, duration=0.6, fs=fs):
    """Body impulse response = sum of decaying modal sinusoids.

    Strike/listen are fractions 0..1 of the plate, mapped to interior matrix
    rows via the shape's index map (so a point always lands on the body, even
    for a masked outline).

    - amp = phi_k(strike) * phi_k(listen): reciprocity. A mode is loud only if
      the strike moves it AND the listening point can see it. Strike or listen
      on a node line of mode k and that mode vanishes.
    - freqs scaled so the lowest mode sits at f0 (the body's lowest resonance).
    - tau = Q / (pi * f) : a resonance of quality Q. Higher modes ring shorter.
    """
    s_node = _frac_to_row(strike, idx_map)
    l_node = _frac_to_row(listen, idx_map)

    t = np.arange(int(fs * duration)) / fs
    freqs = f0 * np.sqrt(evals) / np.sqrt(evals[0])
    y = np.zeros_like(t)
    for k in range(min(n_modes, len(evals))):
        amp = evecs[s_node, k] * evecs[l_node, k]
        tau = Q / (np.pi * freqs[k])
        y += amp * np.sin(2 * np.pi * freqs[k] * t) * np.exp(-t / tau)

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


# ----------------------------------------------------------------------------
# 2b. THE SOUNDHOLE  (top plate + back plate + Helmholtz air, coupled)
# ----------------------------------------------------------------------------
# A real guitar box has three low-frequency degrees of freedom that talk to each
# other through the enclosed air: the top plate flexing, the back plate flexing,
# and the air breathing in/out of the soundhole (a Helmholtz resonator, ~90-110
# Hz). The shared cavity air is a spring linking all three. Coupled oscillators
# repel, so the three bare frequencies split into three new normal modes -- this
# is what gives a guitar its warm low end and its characteristic "boom".

def three_oscillator_model(f_top=180, f_back=200, m_top=0.15, m_back=0.05,
                           Volume=0.012, hole_radius=0.041, hole_thickness=0.003,
                           top_area=0.18, back_area=0.013, rho=1.2, c=343):
    """Solve the 3x3 generalised eigenproblem K phi = w^2 M phi for the coupled
    top/back/air modes. M is the mass matrix; K is the bare stiffnesses plus the
    air-spring term (rho c^2 / V) a a^T, where a = [A_top, A_back, -A_hole] is
    the piston-area vector (air leaves through the hole, hence the minus)."""
    omega_top = 2 * np.pi * f_top
    omega_back = 2 * np.pi * f_back
    hole_area = np.pi * hole_radius ** 2
    L_eff = hole_thickness + 1.2 * hole_radius        # end corrections, both faces
    m_hole = rho * hole_area * L_eff                  # air-plug mass

    a = np.array([top_area, back_area, -hole_area])
    M = np.diag([m_top, m_back, m_hole])
    K = np.diag([m_top * omega_top ** 2, m_back * omega_back ** 2, 0.0]) \
        + (rho * c ** 2 / Volume) * np.outer(a, a)    # shared air spring couples them

    evals, evecs = scipy.linalg.eigh(K, M)
    freqs = np.sqrt(np.clip(evals, 0, None)) / (2 * np.pi)
    return freqs, evecs


def modal_bank(freqs, amps, Qs, t):
    """Sum a set of decaying sinusoids -- one resonance per (freq, amp, Q)."""
    y = np.zeros_like(t)
    for f, a, Q in zip(freqs, amps, Qs):
        tau = Q / (np.pi * f)
        y += a * np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
    return y


def build_body_ir_soundhole(evals, evecs, idx_map, Q=20, strike=(0.5, 0.5),
                            listen=(0.5, 0.5), n_modes=30, f_top=180.0, g_low=1.0,
                            duration=0.6, fs=fs, **osc):
    """Body IR = the three coupled low modes + the higher plate modes, mixed.

    The plate's own fundamental (mode 0) is dropped -- that low resonance is now
    handled properly by the coupled top/back/air model. `g_low` sets how loud the
    coupled low body sits under the plate modes. `**osc` forwards soundhole
    parameters (f_back, Volume, hole_radius, ...) to three_oscillator_model."""
    s_node = _frac_to_row(strike, idx_map)
    l_node = _frac_to_row(listen, idx_map)
    t = np.arange(int(fs * duration)) / fs

    # higher plate modes (skip mode 0 = fundamental, now from the coupled model)
    plate_f = f_top * np.sqrt(evals) / np.sqrt(evals[0])
    freqs_p, amps_p, Qs_p = [], [], []
    for k in range(1, min(n_modes, len(evals))):
        freqs_p.append(plate_f[k])
        amps_p.append(evecs[s_node, k] * evecs[l_node, k])
        Qs_p.append(Q)

    # three coupled low modes (top + back + air)
    freqs3, ev3 = three_oscillator_model(f_top=f_top, **osc)
    amps3 = [ev3[0, k] * (ev3[0, k] + ev3[2, k]) for k in range(3)]   # string drives top
    Qs3 = [30, 30, 15]

    freqs = np.concatenate([freqs3, freqs_p])
    amps = np.concatenate([g_low * np.array(amps3), amps_p])
    Qs = np.concatenate([Qs3, Qs_p])

    y = modal_bank(freqs, amps, Qs, t)
    peak = np.max(np.abs(y))
    return (y / peak) if peak else y


# ----------------------------------------------------------------------------
# 3. THE INSTRUMENT  (string driven through the body)
# ----------------------------------------------------------------------------
# String and body are both linear and time-invariant, so passing the string
# "through" the body is a convolution with the body's impulse response. This is
# a one-way coupling (string -> body). A real bridge is two-way (the body pushes
# back on the string); that feedback is a later milestone.
#
# The body IR is built ONCE per instrument and reused for every note: the body
# does not change between notes, only the string does. So `guitar_voice` takes
# a pre-computed body_IR and just plucks a new string through it. This mirrors
# your notebook's guitar_voice / play_sequence design.

def build_body_ir(ax, ay, resolution, f0, Q, strike, listen, n_modes,
                  model="Plate only", f_back=200.0, volume=0.012,
                  hole_radius=0.041, g_low=1.0, shape="Guitar (figure-8)", fs=fs):
    """Dispatch to the chosen body model on the chosen body shape. `f0` doubles
    as the top-plate resonance f_top in the soundhole model (both scale the
    plate mode set)."""
    evals, evecs, idx_map = plate_modes(ax, ay, int(resolution), shape)   # cached
    if model == "Plate + soundhole":
        return build_body_ir_soundhole(
            evals, evecs, idx_map, Q=Q, strike=strike, listen=listen,
            n_modes=n_modes, f_top=f0, g_low=g_low,
            f_back=f_back, Volume=volume, hole_radius=hole_radius, fs=fs)
    return plate_ir(evals, evecs, idx_map, f0=f0, Q=Q,
                    strike=strike, listen=listen, n_modes=n_modes, fs=fs)


def guitar_voice(f, body_IR, rho=0.99, beta=0.5, fs=fs):
    """One note: pluck a string at pitch f and convolve it through the body."""
    N = note_to_N(f, fs)                        # only used to size the render length
    dur = sustain_duration(rho=rho, N=N, fs=fs)
    s = pluck_string(beta=beta, f=f, duration=dur, rho=rho, fs=fs)  # exact tuning
    note = scipy.signal.fftconvolve(s, body_IR)
    M = min(2000, len(note))
    note[-M:] *= np.linspace(1, 0, M)          # fade the tail so it doesn't click
    return note


def play_sequence(freqs, voice, dt=0.4, fs=fs):
    """

    Note k starts at time k*dt and is SUMMED into the buffer, so the ringing
    tail of each note overlaps the attack of the next. dt < a note's sustain =>
    overlapping/strummed; dt > sustain => separated. The buffer is sized to
    (last onset + full last note) so no tail is clipped even when notes overlap.
    """
    notes = [voice(f) for f in freqs]
    onsets = [round(i * dt * fs) for i in range(len(freqs))]
    total = max(o + len(n) for o, n in zip(onsets, notes))
    out = np.zeros(total)
    for o, n in zip(onsets, notes):
        out[o:o + len(n)] += n
    peak = np.max(np.abs(out))
    return (out / peak) if peak else out     


def make_scale(root_hz, semitones):
    """Equal temperament: f_k = f_root * 2^(k/12)."""
    return root_hz * 2 ** (np.array(semitones) / 12)


def export_pack(lo, hi, beta, rho, ax, ay, resolution, f0, Q,
                sx, sy, lx, ly, n_modes,
                model="Plate only", f_back=200.0, volume=0.012,
                hole_radius=0.041, g_low=0.1, shape="Guitar (figure-8)",
                fname="realsound", fs=fs):
    """Render every chromatic note between lo and hi with the CURRENT settings,
    write each to its own WAV, and zip them into a sample pack.

    The body IR is built once (the instrument is fixed); only the string pitch
    changes per note -- the same build-once structure as guitar_voice. Each note
    is peak-normalised to 0.98 so it's a clean one-shot sample. `fname` is the
    user-chosen base name: the zip is <fname>.zip and each note <fname>_<note>.wav.
    """
    body_IR = build_body_ir(ax, ay, int(resolution), f0, Q,
                            (sx, sy), (lx, ly), int(n_modes),
                            model=model, f_back=f_back, volume=volume,
                            hole_radius=hole_radius, g_low=g_low, shape=shape)
    voice = lambda f: guitar_voice(f, body_IR, rho=rho, beta=beta, fs=fs)

    i0, i1 = ALL_NOTES.index(lo), ALL_NOTES.index(hi)
    if i0 > i1:
        i0, i1 = i1, i0

    base = safe_name(fname)                                # download name is the file's name
    outdir = tempfile.mkdtemp(prefix="realsound_pack_")
    zip_path = os.path.join(outdir, f"{base}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ALL_NOTES[i0:i1 + 1]:
            y = voice(note_name_to_hz(name))
            peak = np.max(np.abs(y))
            if peak:
                y = 0.98 * y / peak
            wav = np.int16(np.clip(y, -1, 1) * 32767)     # 16-bit PCM one-shot
            wav_path = os.path.join(outdir, f"{base}_{name}.wav")
            wavfile.write(wav_path, fs, wav)
            z.write(wav_path, arcname=os.path.basename(wav_path))
    return zip_path


# Equal temperament: f_k = f_root * 2^(k/12).  Used by the scale player.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Full chromatic range used by the sample-pack exporter (C1..B6).
# Defined here, after NOTE_NAMES, because it is built from it.
ALL_NOTES = [f"{n}{o}" for o in range(1, 7) for n in NOTE_NAMES]

# Chords are just interval stacks played together. Feeding these to
# play_sequence with dt=0 stacks them at t=0 (a block chord); a small dt
# staggers the onsets into a strum. (Semitone offsets from the root.)
CHORDS = {
    "Major (1-3-5)":     [0, 4, 7],
    "Minor (1-b3-5)":    [0, 3, 7],
    "Major 7":           [0, 4, 7, 11],
    "Dominant 7":        [0, 4, 7, 10],
    "Minor 7":           [0, 3, 7, 10],
    "Sus4 (1-4-5)":      [0, 5, 7],
    "Power (1-5-8)":     [0, 7, 12],
    "Add octave (1-3-5-8)": [0, 4, 7, 12],
}


def note_name_to_hz(name):
    # e.g. "E2", "A4". A4 = 440 Hz.
    import re
    m = re.match(r"^([A-G]#?)(-?\d)$", name.strip())
    if not m:
        return 220.0
    semis = NOTE_NAMES.index(m.group(1)) + 12 * (int(m.group(2)) + 1)
    a4 = NOTE_NAMES.index("A") + 12 * (4 + 1)
    return 440.0 * 2 ** ((semis - a4) / 12)


# The major scale (semitone offsets) is built inline in the UI's scale player.


# ----------------------------------------------------------------------------
# 4. PLOTTING
# ----------------------------------------------------------------------------
def plot_wave_spectrum(y, title="output"):
    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    t_ms = np.arange(len(y)) / fs * 1000
    n = min(len(y), int(fs * 0.03))                       # first 30 ms of waveform
    ax[0].plot(t_ms[:n], y[:n], lw=0.8)
    ax[0].set_title(f"waveform - {title}")
    ax[0].set_xlabel("ms"); ax[0].set_ylabel("amp")

    X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    ax[1].plot(f, 20 * np.log10(X / (np.max(X) + 1e-12) + 1e-9), lw=0.7)
    ax[1].set_xlim(0, 6000); ax[1].set_ylim(-80, 2)
    ax[1].set_title("spectrum"); ax[1].set_xlabel("Hz"); ax[1].set_ylabel("dB")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# 5. GRADIO UI
# ----------------------------------------------------------------------------
def build_ui():
    import gradio as gr

    def render(note, beta, rho, ax, ay, resolution, f0, Q,
               sx, sy, lx, ly, n_modes, scale_steps, dt, chord_type,
               body_model, f_back, volume, hole_radius, g_low, shape, fname, mode):
        strike = (sx, sy)
        listen = (lx, ly)
        # Build the body IR once, then reuse it for every note via a voice closure.
        body_IR = build_body_ir(ax, ay, int(resolution), f0, Q,
                                 strike, listen, int(n_modes),
                                 model=body_model, f_back=f_back, volume=volume,
                                 hole_radius=hole_radius, g_low=g_low, shape=shape)
        voice = lambda f: guitar_voice(f, body_IR, rho=rho, beta=beta)
        root = note_name_to_hz(note)

        if mode == "Single note":
            y = voice(root)
            peak = np.max(np.abs(y))
            y = (0.98 * y / peak) if peak else y
            title = note
        elif mode == "Chord":
            freqs = make_scale(root, CHORDS[chord_type])   # interval stack
            y = 0.98 * play_sequence(freqs, voice, dt=dt)  # dt=0 -> block; small dt -> strum
            strum = "block" if dt == 0 else f"strum dt={dt:.2f}s"
            title = f"{note} {chord_type} ({strum})"
        else:  # Scale
            semis = [0, 2, 4, 5, 7, 9, 11, 12][:int(scale_steps)]
            freqs = make_scale(root, semis)
            y = 0.98 * play_sequence(freqs, voice, dt=dt)
            title = f"{note} scale ({int(scale_steps)} notes, dt={dt:.2f}s)"
        y = y.astype(np.float32)
        fig = plot_wave_spectrum(y, title)
        # Write a WAV named as the user asked -> that name is what the browser
        # downloads (the audio player's own button can't be renamed).
        wav_path = os.path.join(tempfile.mkdtemp(), f"{safe_name(fname)}.wav")
        wavfile.write(wav_path, fs, np.int16(np.clip(y, -1, 1) * 32767))
        return (fs, y), fig, wav_path

    notes = [f"{n}{o}" for o in range(2, 6) for n in NOTE_NAMES]
    with gr.Blocks(title="RealSound Instrument") as demo:
        gr.Markdown("# RealSound — custom instrument\n"
                    "Every knob is a physical parameter of your string+plate model. "
                    "Adjust, press **Generate**, listen, and use the audio player's "
                    "download button to save the WAV.")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### String (pluck)")
                note = gr.Dropdown(notes, value="E3", label="Note")
                beta = gr.Slider(0.02, 0.5, 0.13, step=0.01,
                                 label="Pluck position β (0=bridge, 0.5=center)")
                rho = gr.Slider(0.9, 0.999, 0.99, step=0.001,
                                label="String decay ρ (higher = longer sustain)")
                gr.Markdown("### Body (plate)")
                ax = gr.Slider(0.5, 8.0, 4.0, step=0.1, label="Stiffness along grain aₓ")
                ay = gr.Slider(0.5, 8.0, 1.0, step=0.1, label="Stiffness across grain a_y")
                resolution = gr.Slider(16, 40, 30, step=2,
                                       label="Grid resolution (higher = more modes, slower)")
                shape = gr.Dropdown(list(MASKS.keys()), value="Guitar (figure-8)",
                                    label="Body shape (outline mask)")
                f0 = gr.Slider(80, 400, 220, step=5,
                               label="Body base freq f₀ / top-plate f_top (Hz)")
                Q = gr.Slider(3, 80, 20, step=1, label="Body Q (resonance sharpness / ring)")

                gr.Markdown("### Body model")
                body_model = gr.Radio(["Plate only", "Plate + soundhole"],
                                      value="Plate only", label="Body model")
                f_back = gr.Slider(120, 320, 200, step=5,
                                   label="Back-plate resonance f_back (Hz) — soundhole")
                volume = gr.Slider(0.004, 0.025, 0.012, step=0.001,
                                   label="Cavity volume V (m³) — lower = higher air pitch")
                hole_radius = gr.Slider(0.02, 0.06, 0.041, step=0.001,
                                        label="Soundhole radius (m) — sets Helmholtz air pitch")
                g_low = gr.Slider(0.0, 2.0, 0.1, step=0.05,
                                  label="Low-body gain g_low — keep low so the shape isn't buried")
            with gr.Column():
                gr.Markdown("### Strike & listen points (fractions of the plate)")
                sx = gr.Slider(0, 1, 0.5, step=0.02, label="Strike x")
                sy = gr.Slider(0, 1, 0.5, step=0.02, label="Strike y")
                lx = gr.Slider(0, 1, 0.5, step=0.02, label="Listen x")
                ly = gr.Slider(0, 1, 0.5, step=0.02, label="Listen y")
                n_modes = gr.Slider(1, 60, 12, step=1, label="Number of modes")
                mode = gr.Radio(["Single note", "Chord", "Scale"],
                                value="Single note", label="Play")
                chord_type = gr.Dropdown(list(CHORDS.keys()), value="Major (1-3-5)",
                                         label="Chord type (Chord mode)")
                scale_steps = gr.Slider(2, 8, 8, step=1, label="Scale length (notes)")
                dt = gr.Slider(0.0, 1.5, 0.4, step=0.02,
                               label="Note spacing dt (s) — 0 = block chord, small = strum")
                fname = gr.Textbox(value="realsound",
                                   label="File name (no extension) — used for downloads")
                go = gr.Button("Generate", variant="primary")
                audio = gr.Audio(label="Sound (listen)", type="numpy")
                download = gr.DownloadButton("Download WAV (named)")
                plot = gr.Plot(label="Waveform + spectrum")

                gr.Markdown("### Export sample pack")
                with gr.Row():
                    lo_note = gr.Dropdown(ALL_NOTES, value="E2", label="Lowest note")
                    hi_note = gr.Dropdown(ALL_NOTES, value="E5", label="Highest note")
                export_btn = gr.Button("Download all notes (.zip)")
                pack_file = gr.File(label="Sample pack — <name>_<note>.wav in <name>.zip")

        inputs = [note, beta, rho, ax, ay, resolution, f0, Q,
                  sx, sy, lx, ly, n_modes, scale_steps, dt, chord_type,
                  body_model, f_back, volume, hole_radius, g_low, shape, fname, mode]
        go.click(render, inputs=inputs, outputs=[audio, plot, download])

        export_btn.click(
            export_pack,
            inputs=[lo_note, hi_note, beta, rho, ax, ay, resolution, f0, Q,
                    sx, sy, lx, ly, n_modes,
                    body_model, f_back, volume, hole_radius, g_low, shape, fname],
            outputs=pack_file,
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
