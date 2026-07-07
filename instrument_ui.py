"""
instrument_ui.py  --  Interactive UI for the RealSound custom-instrument generator.

This is *scaffolding* around the physics engine you built in guitar.ipynb.
Nothing here invents new DSP: every sound is produced by your own functions
(pluck_string -> orthotropic_plate -> real_plate_strike -> fftconvolve).
The UI just turns the function arguments into sliders, plays the result, and
lets you download the WAV.

Run it with:
    pip install gradio        # one-time
    python instrument_ui.py
then open the http://127.0.0.1:7860 link it prints.

See RealSound_Instrument_UI_Explainer.md for the physics behind each control
and the design decisions (including the indexing bug this file fixes).
"""

import functools
import numpy as np
import scipy.linalg
import scipy.signal
import matplotlib
matplotlib.use("Agg")            # headless: we render figures to images, never a window
import matplotlib.pyplot as plt

fs = 44100                       # sample rate (Hz) -- CD standard, as in the notebook


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
    """Invert f ~= fs/(N+0.5) to get the delay-line length for a pitch.

    Physical note: N must be an integer number of samples, so the achievable
    pitches are quantised. The error grows at high frequency (small N), where
    one sample is a large fraction of the period. Below we accept that error;
    a fractional-delay allpass would fix it, and is a natural later milestone.
    """
    N = int(round(fs / freq_hz - 0.5))
    return max(2, N)


def pluck_string(beta=0.5, N=100, duration=2.0, fs=fs, rho=1.0):
    """Your notebook function, unchanged."""
    p = int(beta * N)
    p = min(max(p, 1), N - 1)                 # keep the pluck strictly inside the string
    increasing = np.linspace(0, 1, p)
    decreasing = np.linspace(1, 0, N - p)
    buf = np.concatenate((increasing, decreasing))   # triangular pluck shape

    out = np.zeros(int(fs * duration))
    idx = 0
    last = 0.0
    for i in range(len(out)):
        out[i] = buf[idx]
        x = buf[idx]
        buf[idx] = 0.5 * (x + last) * rho     # 2-point average + loss -> low-pass decay
        last = x
        idx = (idx + 1) % N
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

@functools.lru_cache(maxsize=16)
def orthotropic_plate(ax=4.0, ay=1.0, resolution=30):
    """Build the plate operator and diagonalise it. Cached by its arguments.

    Returns (evals, evecs, resolution). ax=ay reproduces the isotropic plate."""
    res = int(resolution)
    Nn = res * res

    def idx(i, j):
        return i * res + j

    L = np.zeros((Nn, Nn))
    for i in range(res):
        for j in range(res):
            p = idx(i, j)
            L[p, p] = -(2 * ax + 2 * ay)                 # anisotropic Laplacian stencil
            if i > 0:        L[p, idx(i - 1, j)] = ax
            if i < res - 1:  L[p, idx(i + 1, j)] = ax
            if j > 0:        L[p, idx(i, j - 1)] = ay
            if j < res - 1:  L[p, idx(i, j + 1)] = ay
    K = L @ L                                            # plate = Laplacian squared
    evals, evecs = scipy.linalg.eigh(K)                 # modes: K phi = lambda phi
    return evals, evecs, res


def plate_ir(evals, evecs, res, f0=220.0, Q=20.0,
             strike=(0.5, 0.5), listen=(0.5, 0.5),
             n_modes=10, duration=0.6, fs=fs):
    """Body impulse response = sum of decaying modal sinusoids.

    IMPORTANT FIX vs. the notebook: the original real_plate_strike used a
    *global* idx() built for a 20x20 grid, so at resolution=40 the strike and
    listen points landed on the wrong nodes. Here idx is rebuilt from the
    actual resolution, and strike/listen are given as fractions 0..1 of the
    plate so they mean the same physical spot at any resolution.

    - amp = phi_k(strike) * phi_k(listen): reciprocity. A mode is loud only if
      the strike moves it AND the listening point can see it. Strike or listen
      on a node line of mode k and that mode vanishes.
    - freqs scaled so the lowest mode sits at f0 (the body's lowest resonance).
    - tau = Q / (pi * f) : a resonance of quality Q. Higher modes ring shorter.
    """
    res = int(res)

    def idx(i, j):
        return i * res + j

    def frac_to_node(pt):
        i = min(max(int(round(pt[0] * (res - 1))), 0), res - 1)
        j = min(max(int(round(pt[1] * (res - 1))), 0), res - 1)
        return idx(i, j)

    s_node = frac_to_node(strike)
    l_node = frac_to_node(listen)

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

def build_body_ir(ax, ay, resolution, f0, Q, strike, listen, n_modes, fs=fs):
    evals, evecs, res = orthotropic_plate(ax, ay, resolution)   # cached
    return plate_ir(evals, evecs, res, f0=f0, Q=Q,
                    strike=strike, listen=listen, n_modes=n_modes, fs=fs)


def guitar_voice(f, body_IR, rho=0.99, beta=0.5, fs=fs):
    """One note: pluck a string at pitch f and convolve it through the body."""
    N = note_to_N(f, fs)                       # invert the KS pitch law for line length
    dur = sustain_duration(rho=rho, N=N, fs=fs)
    s = pluck_string(beta=beta, N=N, duration=dur, rho=rho, fs=fs)
    note = scipy.signal.fftconvolve(s, body_IR)
    M = min(2000, len(note))
    note[-M:] *= np.linspace(1, 0, M)          # fade the tail so it doesn't click
    return note


def play_sequence(freqs, voice, dt=0.4, fs=fs):
    """Your onset-mixing sequencer.

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
    return (out / peak) if peak else out       # (fixed: original paste dropped a paren)


def make_scale(root_hz, semitones):
    """Equal temperament: f_k = f_root * 2^(k/12)."""
    return root_hz * 2 ** (np.array(semitones) / 12)


# Equal temperament: f_k = f_root * 2^(k/12).  Used by the scale player.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
               sx, sy, lx, ly, n_modes, scale_steps, dt, chord_type, mode):
        strike = (sx, sy)
        listen = (lx, ly)
        # Build the body IR once, then reuse it for every note via a voice closure.
        body_IR = build_body_ir(ax, ay, int(resolution), f0, Q,
                                 strike, listen, int(n_modes))
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
        return (fs, y), fig

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
                f0 = gr.Slider(80, 400, 220, step=5, label="Body base frequency f₀ (Hz)")
                Q = gr.Slider(3, 80, 20, step=1, label="Body Q (resonance sharpness / ring)")
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
                go = gr.Button("Generate", variant="primary")
                audio = gr.Audio(label="Sound (download from the player)", type="numpy")
                plot = gr.Plot(label="Waveform + spectrum")

        inputs = [note, beta, rho, ax, ay, resolution, f0, Q,
                  sx, sy, lx, ly, n_modes, scale_steps, dt, chord_type, mode]
        go.click(render, inputs=inputs, outputs=[audio, plot])
    return demo


if __name__ == "__main__":
    build_ui().launch()
