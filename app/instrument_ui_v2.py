"""RealSound — instrument panel (NiceGUI). Every function in realsound.py, plus
preset save/load, sample-pack export, and a live view of the body you designed.

Run:  python app/panel.py     then open http://localhost:5500

Layout map:
    1. SETTINGS      colours, note table, patterns        -- look & feel
    2. ENGINE GLUE   every realsound.py call, cached      -- physics wiring
    3. VIEWS         mask preview, waveform + spectrum    -- the right column
    4. CONTROLS      the sliders, grouped in sections     -- the left column
    5. PRESETS       collect/apply every control's value  -- save & load
    6. ACTION        Play, download, export, live redraw  -- behaviour
See "MODIFYING" at the bottom.

Every control made with knob() or pick() registers itself in CONTROLS, so a new
knob is automatically saved, loaded and exported. Nothing else to update.
"""
import io, json, sys, tempfile, uuid, zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')                      # headless: figures render to images
import matplotlib.pyplot as plt
from nicegui import ui

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import realsound as rs

# ============================================================ 1. SETTINGS ====
FS = 44100
INK, MUTED = 'text-ink', 'text-ink opacity-60'
CARD = 'w-[30rem] bg-panel gap-2'          # left column (controls)
VIEW = 'w-[26rem] bg-panel gap-2'          # right column (output)
SEC = 'w-full'
FG, BG = '#eef3f0', '#10222e'              # plot colours, matched to the cards

_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES = {f'{n}{o}': 440.0 * 2 ** ((_NAMES.index(n) + 12 * (o + 1) - 69) / 12)
         for o in range(2, 6) for n in _NAMES}

PATTERNS = {
    'Major triad': [0, 4, 7],
    'Minor triad': [0, 3, 7],
    'Major 7th':   [0, 4, 7, 11],
    'Major scale': [0, 2, 4, 5, 7, 9, 11, 12],
    'Minor scale': [0, 2, 3, 5, 7, 8, 10, 12],
}

ui.colors(primary='#7fb069', panel='#10222e', ink='#eef3f0')
ui.dark_mode(value=True)
OUT = Path(tempfile.mkdtemp(prefix='realsound_'))


# ========================================================= 2. ENGINE GLUE ====
def current_mask(shape, res, rTop, rBot, cyTop, cyBot, rHole):
    """The body outline only -- cheap, no eigenproblem. Used by the preview."""
    if shape == 'Square':
        return np.ones((res, res), dtype=bool)
    return rs.boolean_8_grid(res=res, rTop=rTop, rBot=rBot,
                             cyTop=cyTop, cyBot=cyBot, rHole=rHole)


@lru_cache(maxsize=16)
def solve_body(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole):
    """Body geometry -> modes. THE slow step, cached on exactly what it uses."""
    if shape == 'Square':
        _K, evals, evecs = rs.orthotropic_plate(ax=ax, ay=ay, resolution=res)
        return evals, evecs, np.arange(res * res).reshape(res, res)
    mask = current_mask(shape, res, rTop, rBot, cyTop, cyBot, rHole)
    return rs.plate_from_mask(mask, ax=ax, ay=ay)


snap_inside = rs.snap_inside          # graduated into realsound.py


@lru_cache(maxsize=16)
def body_ir(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole,
            n_modes, Q, f_top, g_low, sxf, syf, lxf, lyf,
            f_back, volume, hole_radius, m_top, m_back,
            radiated=False, ir_sec=0.6):
    """Modes -> impulse response, via rs.build_body_IR (which runs modal_bank
    and three_oscillator_model inside; cavity params ride along via **osc)."""
    evals, evecs, idx = solve_body(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole)
    sx, sy = snap_inside(idx >= 0, sxf, syf)
    lx, ly = snap_inside(idx >= 0, lxf, lyf)
    t = np.arange(int(FS * ir_sec)) / FS
    return rs.build_body_IR(evals, evecs, idx, t, n_modes=n_modes, Q=Q,
                            sx=sx, sy=sy, lx=lx, ly=ly,
                            f_top=f_top, g_low=g_low, radiated=radiated,
                            f_back=f_back, Volume=volume, hole_radius=hole_radius,
                            m_top=m_top, m_back=m_back)


def make_voice(ir, beta, rho, duration):
    """f -> audio, ONE-WAY: string excites body, body cannot push back."""
    def voice(f):
        s = rs.pluck_string(f=f, beta=beta, duration=duration, fs=FS, rho=rho)
        n = scipy.signal.fftconvolve(s, ir)
        n[-2000:] *= np.linspace(1, 0, 2000)
        return n
    return voice


# --- TWO-WAY COUPLING lives in realsound.py (BodyAdmittance, coupled_string,
# wolf_body). A convolved IR is a finished recording and cannot push back; the
# two-way path runs the body as live per-sample resonators instead.

def make_voice_twoway(modes, beta, rho, duration, k, use_wolf, wolf_det,
                      wolf_Q, wolf_amp, n_modes, Q, f_top, cells, radiated):
    """f -> audio, TWO-WAY. A fresh body per note: the resonators are stateful,
    and the wolf mode is placed relative to the note being played."""
    evals, evecs, idx = modes
    sx, sy, lx, ly = cells

    def voice(f):
        body = rs.wolf_body(evals, evecs, idx,
                            wolf_f=f * (1 + wolf_det) if use_wolf else None,
                            k=k, wolf_Q=wolf_Q, wolf_amp=wolf_amp,
                            n_modes=n_modes, Q=Q, f_top=f_top,
                            sx=sx, sy=sy, lx=lx, ly=ly,
                            radiated=radiated, fs=FS)
        y = rs.coupled_string(f, body, beta=beta, duration=duration, fs=FS, rho=rho)
        y[-2000:] *= np.linspace(1, 0, 2000)
        return y
    return voice


# --- ELECTRIC ---------------------------------------------------------------
# An electric guitar is not a second instrument: a solid body is a bridge that
# takes almost no energy and returns none, and the pickup's position comb is the
# listen point under another name. What is genuinely new is the transducer
# (velocity sensing + an electrical resonance) and the fact that a nonlinearity
# now sits MID-CHAIN, so the stages can no longer be permuted -- the commuted
# synthesis that makes the acoustic body a single convolution is dead here.
# Chain: string -> velocity -> position comb -> RLC resonance -> drive -> cab.

@lru_cache(maxsize=8)
def cab_ir(f_lo, f_hi):
    """Speaker + box as an IR. Linear, so it convolves exactly like the acoustic
    body -- the electric chain IS the acoustic chain with the body IR swapped."""
    return rs.speaker_cab_ir(fs=FS, f_lo=f_lo, f_hi=f_hi)


def make_voice_electric(beta, rho, duration, b, alpha, beta_p, kind,
                        shape, drive, bias, factor, use_cab, f_lo, f_hi):
    """f -> audio, ELECTRIC. Evaluated in order; do not reorder for speed."""
    ir = cab_ir(f_lo, f_hi) if use_cab else None
    def voice(f):
        return rs.electric_guitar(
            f=f, duration=duration, fs=FS,
            beta=beta, rho=rho, b=b, alpha=alpha,
            beta_p=beta_p, pickup_kind=kind,
            drive=drive, bias=bias, shape=shape, factor=int(factor),
            cab=use_cab, cab_ir=ir, normalize=True)
    return voice


def norm16(y):
    """Peak-normalise to 16-bit. Guarded: a fully silent buffer is legal here
    (two-way with k=0 radiates nothing), and dividing by its zero peak would
    produce NaN."""
    peak = np.max(np.abs(y))
    if not np.isfinite(peak) or peak < 1e-12:
        return np.zeros(len(y), dtype=np.int16)
    return np.int16(0.98 * y / peak * 32767)


# ============================================================== 3. VIEWS =====
def style_axes(*axes):
    for a in axes:
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(FG); sp.set_alpha(0.3)
        a.tick_params(colors=FG, labelsize=7)
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG)
        a.title.set_color(FG)


def draw_mask():
    """The body outline, with strike (x) and listen (o) marked. Cheap: this is
    the boolean grid only, so it can redraw live while a slider is dragged."""
    mask = current_mask(shape_sel.value, int(res.value), rTop.value, rBot.value,
                        cyTop.value, cyBot.value, rHole.value)
    sx, sy = snap_inside(mask, sxf.value, syf.value)
    lx, ly = snap_inside(mask, lxf.value, lyf.value)
    mask_box.clear()
    with mask_box:
        with ui.pyplot(figsize=(3.4, 3.4), close=True):
            fig = plt.gcf(); fig.patch.set_facecolor(BG)
            a = plt.gca(); style_axes(a)
            field = np.where(mask, 1.0, np.nan)   # NaN outside -> card shows through
            a.imshow(field.T, origin='lower', cmap='Greens', vmin=0, vmax=1.6)
            a.plot(sx, sy, 'x', color='#ff6b6b', ms=9, mew=2, label='strike')
            if radiate.value:                     # whole surface is the listener
                ii, jj = np.where(mask)
                a.plot(ii, jj, 's', color='#4dabf7', ms=1.5, alpha=0.5, label='radiating')
            else:
                a.plot(lx, ly, 'o', color='#4dabf7', ms=8, mfc='none', mew=2, label='listen')
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f'{int(mask.sum())} cells', fontsize=8)
            a.legend(fontsize=7, facecolor=BG, edgecolor='none',
                     labelcolor=FG, loc='lower right')


def draw_signal(y):
    """Waveform (first 30 ms) and spectrum in dB -- verify by eye."""
    plot_box.clear()
    with plot_box:
        with ui.pyplot(figsize=(3.4, 3.2), close=True):
            fig = plt.gcf(); fig.patch.set_facecolor(BG)
            a1 = plt.subplot(2, 1, 1); a2 = plt.subplot(2, 1, 2)
            style_axes(a1, a2)
            n = min(len(y), int(FS * 0.03))
            a1.plot(np.arange(n) / FS * 1000, y[:n], lw=0.7, color='#7fb069')
            a1.set_title('waveform', fontsize=8); a1.set_xlabel('ms', fontsize=7)
            X = np.abs(np.fft.rfft(y * np.hanning(len(y))))
            f = np.fft.rfftfreq(len(y), 1 / FS)
            a2.plot(f, 20 * np.log10(X / (np.max(X) + 1e-12) + 1e-9),
                    lw=0.6, color='#4dabf7')
            a2.set_xlim(0, 4000); a2.set_ylim(-80, 2)
            a2.set_title('spectrum', fontsize=8); a2.set_xlabel('Hz', fontsize=7)
            plt.tight_layout()


# =========================================================== 4. CONTROLS =====
CONTROLS = {}          # name -> element. Anything in here is saved/loaded.


def knob(key, label, lo, hi, val, step, fmt='{:.2f}'):
    with ui.row().classes('w-full items-center no-wrap'):
        ui.label(label).classes(f'{MUTED} text-sm w-36')
        s = ui.slider(min=lo, max=hi, value=val, step=step) \
              .props('color=primary').classes('grow')
        ui.label().classes(f'{INK} text-sm w-14 text-right') \
          .bind_text_from(s, 'value', lambda v: fmt.format(v))
    CONTROLS[key] = s
    return s


def pick(key, options, value, label):
    s = ui.select(options, value=value, label=label) \
          .props('dark outlined dense').classes('grow')
    CONTROLS[key] = s
    return s


def flag(key, label, value=False):
    s = ui.switch(label, value=value).props('color=primary').classes('w-full')
    s._classes.append(INK)
    CONTROLS[key] = s
    return s


with ui.row().classes('w-full justify-center items-start gap-4 p-2'):

    with ui.card().classes(CARD):                       # ---------- left ----
        ui.label('RealSound').classes(f'{INK} text-lg')

        with ui.row().classes('w-full items-center no-wrap'):
            note_sel = pick('root', list(NOTES), 'E2', 'Root')
            mode_sel = pick('mode', ['Note', 'Chord', 'Scale'], 'Note', 'Play')

        inst_sel = pick('instrument', ['Acoustic', 'Electric'], 'Acoustic',
                        'Instrument')

        with ui.expansion('String', value=True).classes(SEC):
            beta = knob('beta', 'Pluck position', 0.02, 0.50, 0.13, 0.01)
            rho = knob('rho', 'Sustain', 0.90, 0.9999, 0.99, 0.0001, '{:.4f}')
            dur = knob('duration', 'Note length (s)', 0.5, 4.0, 2.0, 0.1, '{:.1f}')

        with ui.expansion('Body shape').classes(SEC) as sec_shape:
            shape_sel = pick('shape', ['Figure-8', 'Square'], 'Figure-8', 'Outline')
            res = knob('res', 'Grid resolution', 16, 44, 32, 2, '{:.0f}')
            rTop = knob('rTop', 'Upper bout radius', 0.08, 0.30, 0.18, 0.01)
            rBot = knob('rBot', 'Lower bout radius', 0.10, 0.35, 0.25, 0.01)
            cyTop = knob('cyTop', 'Upper bout centre', 0.50, 0.90, 0.68, 0.01)
            cyBot = knob('cyBot', 'Lower bout centre', 0.10, 0.50, 0.35, 0.01)
            rHole = knob('rHole', 'Soundhole radius', 0.00, 0.12, 0.05, 0.01)
            for k in (rTop, rBot, cyTop, cyBot, rHole):
                k.bind_visibility_from(shape_sel, 'value', lambda v: v == 'Figure-8')

        with ui.expansion('Plate').classes(SEC) as sec_plate:
            ax = knob('ax', 'Stiffness along grain', 0.5, 8.0, 4.0, 0.1, '{:.1f}')
            ay = knob('ay', 'Stiffness across grain', 0.5, 8.0, 1.0, 0.1, '{:.1f}')
            f_top = knob('f_top', 'Top resonance (Hz)', 80, 400, 180, 5, '{:.0f}')
            Q = knob('Q', 'Resonance Q', 3, 60, 20, 1, '{:.0f}')
            n_modes = knob('n_modes', 'Number of modes', 4, 60, 30, 1, '{:.0f}')
            g_low = knob('g_low', 'Low-end boom', 0.0, 1.0, 0.1, 0.05)

        with ui.expansion('Strike and listen').classes(SEC) as sec_strike:
            sxf = knob('sx', 'Strike x', 0.0, 1.0, 0.35, 0.05)
            syf = knob('sy', 'Strike y', 0.0, 1.0, 0.35, 0.05)
            radiate = flag('radiated', 'Radiated (whole surface)')
            lxf = knob('lx', 'Listen x', 0.0, 1.0, 0.60, 0.05)
            lyf = knob('ly', 'Listen y', 0.0, 1.0, 0.60, 0.05)
            for k in (lxf, lyf):                      # no single point when radiating
                k.bind_visibility_from(radiate, 'value', lambda v: not v)

        with ui.expansion('Coupling').classes(SEC) as sec_coupling:
            coupling = pick('coupling', ['One-way (IR)', 'Two-way (moving bridge)'],
                            'One-way (IR)', 'String to body')
            ui.label('Two-way lets the body push back on the string: wolf notes, '
                     'dead spots, mode repulsion. Costs ~0.8 s per 3 s note.') \
              .classes(f'{MUTED} text-xs')
            # k = Zs*Y at resonance. Physical range ~0.002-0.05; k=1 would be a
            # perfectly matched load. Starts at 0.002 because the output is
            # bridge velocity, so k=0 is exactly silent.
            kc = knob('k', 'Bridge coupling k', 0.002, 0.20, 0.03, 0.002, '{:.3f}')
            wolf_on = flag('wolf', 'Add a wolf mode', True)
            wolf_det = knob('wolf_detune', 'Wolf detune (%)', -20.0, 20.0, 0.0, 0.5, '{:+.1f}')
            wolf_Q = knob('wolf_Q', 'Wolf mode Q', 20, 600, 120, 10, '{:.0f}')
            wolf_amp = knob('wolf_amp', 'Wolf mode strength', 0.1, 3.0, 1.0, 0.1, '{:.1f}')
            for w in (kc, wolf_on, wolf_det, wolf_Q, wolf_amp):
                w.bind_visibility_from(coupling, 'value', lambda v: v.startswith('Two'))
            for w in (wolf_det, wolf_Q, wolf_amp):
                w.bind_visibility_from(wolf_on, 'value')

        with ui.expansion('Air cavity').classes(SEC) as sec_cavity:
            ui.label('One-way only: the two-way path uses the live plate bank.') \
              .classes(f'{MUTED} text-xs')
            f_back = knob('f_back', 'Back resonance (Hz)', 120, 320, 200, 5, '{:.0f}')
            volume = knob('volume', 'Cavity volume (m³)', 0.004, 0.025, 0.012, 0.001, '{:.3f}')
            hole_r = knob('hole_radius', 'Soundhole radius (m)', 0.02, 0.06, 0.041, 0.001, '{:.3f}')
            m_top = knob('m_top', 'Top plate mass', 0.05, 0.40, 0.15, 0.01)
            m_back = knob('m_back', 'Back plate mass', 0.02, 0.20, 0.05, 0.01)

        # ----- electric-only sections -----
        with ui.expansion('Electric string').classes(SEC) as sec_estring:
            ui.label('Tension modulation: a hard attack starts sharp and falls '
                     'to pitch. Not a waveshaper — the nonlinearity is in the '
                     'delay length, so it moves f0 instead of adding harmonics.') \
              .classes(f'{MUTED} text-xs')
            alpha = knob('alpha', 'Tension depth α', 0.0, 20.0, 0.0, 0.5, '{:.1f}')
            bfilt = knob('b', 'Loop filter b', 0.02, 0.60, 0.10, 0.02)

        with ui.expansion('Pickup').classes(SEC) as sec_pickup:
            pu_kind = pick('pickup_kind', list(rs.PICKUPS), 'single', 'Type')
            beta_p = knob('beta_p', 'Position (from bridge)', 0.04, 0.30, 0.08, 0.01)
            ui.label('Comb nulls every f0/β_p: 0.25 (neck) nulls every 4th '
                     'harmonic, 0.08 (bridge) every 12th.') \
              .classes(f'{MUTED} text-xs')

        with ui.expansion('Overdrive').classes(SEC) as sec_drive:
            shape_nl = pick('nl_shape', list(rs.SHAPES), 'tanh', 'Curve')
            # Empirical: the pickup delivers ~0.045 peak, so the useful drive
            # range here is ~20x higher than nonlinear()'s documented 1-8.
            # Crest factor falls 5.1 -> 2.5 between drive 10 and 100, then flat.
            drive = knob('drive', 'Drive', 0, 300, 0, 5, '{:.0f}')
            bias = knob('bias', 'Bias (asymmetry)', 0.0, 0.8, 0.0, 0.05)
            factor = knob('factor', 'Oversampling', 4, 32, 8, 4, '{:.0f}')
            ui.label('Bias breaks the odd symmetry and brings in even harmonics '
                     '— it changes the sound more than the curve does. 0 = clean '
                     'bypass. Corners (hard/fold) need more oversampling.') \
              .classes(f'{MUTED} text-xs')

        with ui.expansion('Speaker cab').classes(SEC) as sec_cab:
            use_cab = flag('cab', 'Speaker cab', True)
            cab_lo = knob('cab_lo', 'Cone resonance (Hz)', 60, 160, 90, 5, '{:.0f}')
            cab_hi = knob('cab_hi', 'Cone mass rolloff (Hz)', 2000, 7000, 4200, 100, '{:.0f}')
            for w in (cab_lo, cab_hi):
                w.bind_visibility_from(use_cab, 'value')

        with ui.expansion('Sequence').classes(SEC):
            patt_sel = pick('pattern', list(PATTERNS), 'Major triad', 'Pattern')
            dt = knob('dt', 'Note spacing (s)', 0.0, 1.0, 0.35, 0.05)
        for w in (patt_sel, dt):
            w.bind_visibility_from(mode_sel, 'value', lambda v: v != 'Note')

        # one selector decides which half of the panel exists
        for s in (sec_shape, sec_plate, sec_strike, sec_coupling, sec_cavity):
            s.bind_visibility_from(inst_sel, 'value', lambda v: v == 'Acoustic')
        for s in (sec_estring, sec_pickup, sec_drive, sec_cab):
            s.bind_visibility_from(inst_sel, 'value', lambda v: v == 'Electric')

    with ui.card().classes(VIEW):                        # ---------- right ---
        body_hdr = ui.label('Body').classes(f'{INK} text-lg')
        mask_box = ui.column().classes('w-full items-center')
        for w in (body_hdr, mask_box):       # no plate to draw for a solid body
            w.bind_visibility_from(inst_sel, 'value', lambda v: v == 'Acoustic')

        play = ui.button('Play').props('color=primary').classes('w-full')
        tuning = ui.label().classes(f'{MUTED} text-xs')
        spectrum = ui.label().classes(f'{MUTED} text-xs')
        out = ui.column().classes('w-full')
        plot_box = ui.column().classes('w-full items-center')

        with ui.expansion('Instrument file').classes(SEC):
            name_in = ui.input('Instrument name', value='my instrument') \
                        .props('dark outlined dense').classes('w-full')
            with ui.row().classes('w-full no-wrap gap-2'):
                ui.button('Save .json', on_click=lambda: save_preset()) \
                  .props('outline color=primary').classes('grow')
                ui.button('Download .wav', on_click=lambda: save_wav()) \
                  .props('outline color=primary').classes('grow')
            ui.upload(label='Load .json', auto_upload=True,
                      on_upload=lambda e: load_preset(e)).props('dark').classes('w-full')

        with ui.expansion('Sample pack').classes(SEC):
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                lo_sel = ui.select(list(NOTES), value='E2', label='From') \
                           .props('dark outlined dense').classes('grow')
                hi_sel = ui.select(list(NOTES), value='E4', label='To') \
                           .props('dark outlined dense').classes('grow')
            ui.label('One .wav per semitone, zipped with the preset.') \
              .classes(f'{MUTED} text-xs')
            ui.button('Export pack .zip', on_click=lambda: export_pack()) \
              .props('outline color=primary').classes('w-full')


# ============================================================ 5. PRESETS =====
def collect():
    """Every control's value -> a plain dict. This IS the instrument."""
    return {k: el.value for k, el in CONTROLS.items()}


def apply(data):
    """Set every control from a dict; unknown keys ignored, missing keys kept."""
    for k, v in data.items():
        if k in CONTROLS:
            CONTROLS[k].value = v


def slug(s):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in s) or 'instrument'


def save_preset():
    doc = {'name': name_in.value, 'params': collect()}
    ui.download.content(json.dumps(doc, indent=2),
                        f'{slug(name_in.value)}.json', 'application/json')


async def load_preset(e):
    try:
        doc = await e.file.json()                  # FileUpload.json() is async
        apply(doc.get('params', doc))              # accept a bare params dict too
        if 'name' in doc:
            name_in.value = doc['name']
        draw_mask()
        ui.notify(f"Loaded {doc.get('name', 'instrument')}", color='positive')
    except Exception as err:
        ui.notify(f'Could not read that file: {err}', color='negative')


# ============================================================= 6. ACTION =====
def current_ir():
    return body_ir(shape_sel.value, int(res.value), ax.value, ay.value,
                   rTop.value, rBot.value, cyTop.value, cyBot.value, rHole.value,
                   int(n_modes.value), Q.value, f_top.value, g_low.value,
                   sxf.value, syf.value, lxf.value, lyf.value,
                   f_back.value, volume.value, hole_r.value,
                   m_top.value, m_back.value, radiate.value)


def current_voice():
    """The instrument: f -> audio. Electric runs the transducer chain; acoustic
    either convolves a precomputed IR (one-way) or runs a live resonator bank
    the string can push against (two-way)."""
    if inst_sel.value == 'Electric':
        return make_voice_electric(
            beta.value, rho.value, dur.value, bfilt.value, alpha.value,
            beta_p.value, pu_kind.value, shape_nl.value, drive.value,
            bias.value, factor.value, use_cab.value, cab_lo.value, cab_hi.value)
    if coupling.value.startswith('Two'):
        modes = solve_body(shape_sel.value, int(res.value), ax.value, ay.value,
                           rTop.value, rBot.value, cyTop.value, cyBot.value,
                           rHole.value)
        idx = modes[2]
        sx, sy = snap_inside(idx >= 0, sxf.value, syf.value)
        lx, ly = snap_inside(idx >= 0, lxf.value, lyf.value)
        return make_voice_twoway(
            modes, beta.value, rho.value, dur.value, kc.value,
            wolf_on.value, wolf_det.value / 100.0, wolf_Q.value, wolf_amp.value,
            int(n_modes.value), Q.value, f_top.value,
            (sx, sy, lx, ly), radiate.value)
    return make_voice(current_ir(), beta.value, rho.value, dur.value)


def render_current():
    """Whatever Play would produce, as float audio."""
    voice = current_voice()
    root = NOTES[note_sel.value]
    if mode_sel.value == 'Note':
        return voice(root)
    freqs = rs.make_scale(root, PATTERNS[patt_sel.value])
    gap = 0.0 if mode_sel.value == 'Chord' else dt.value
    return rs.play_sequence(freqs, voice, dt=gap, fs=FS)


def on_play():
    play.disable()
    try:
        y = render_current()
        path = OUT / f'{uuid.uuid4().hex[:8]}.wav'     # new name defeats caching
        wavfile.write(path, FS, norm16(y))
        root = NOTES[note_sel.value]
        raw = rs.pluck_string(f=root, beta=beta.value, duration=1.0, fs=FS, rho=rho.value)
        meas = rs.measure_frequency(raw, fs=FS)
        tuning.text = (f'target {root:.2f} Hz · measured {meas:.2f} Hz · '
                       f'{1200 * np.log2(meas / root):+.1f} cents')
        # Electric: report what the nonlinearity actually did. Harmonics are the
        # point; inharmonic energy is the damage.
        if inst_sel.value == 'Electric' and mode_sel.value == 'Note':
            lv, inh = rs.harmonic_report(y, root, fs=FS)
            if np.isfinite(inh):
                spectrum.text = ('h2 {:+.0f}  h3 {:+.0f}  h5 {:+.0f} dB · '
                                 'inharmonic {:.0f} dB'.format(lv[1], lv[2], lv[4], inh))
        else:
            spectrum.text = ''
        out.clear()
        with out:
            ui.audio(str(path), autoplay=True).classes('w-full')
        draw_signal(y)
    finally:
        play.enable()


def save_wav():
    buf = io.BytesIO()
    wavfile.write(buf, FS, norm16(render_current()))
    ui.download.content(buf.getvalue(), f'{slug(name_in.value)}.wav', 'audio/wav')


def export_pack():
    names = list(NOTES)
    i0, i1 = names.index(lo_sel.value), names.index(hi_sel.value)
    if i0 > i1:
        i0, i1 = i1, i0
    chosen = names[i0:i1 + 1]
    slow = coupling.value.startswith('Two')
    ui.notify(f'Rendering {len(chosen)} notes…'
              + (' (two-way: ~1 s each)' if slow else ''))
    voice = current_voice()
    base = slug(name_in.value)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as z:
        for nm in chosen:
            wbuf = io.BytesIO()
            wavfile.write(wbuf, FS, norm16(voice(NOTES[nm])))
            z.writestr(f'{base}/{base}_{nm}.wav', wbuf.getvalue())
        z.writestr(f'{base}/{base}.json',
                   json.dumps({'name': name_in.value, 'params': collect()}, indent=2))
    ui.download.content(zbuf.getvalue(), f'{base}_pack.zip', 'application/zip')
    ui.notify('Pack ready', color='positive')


play.on_click(on_play)

# live preview: redraw the outline whenever a geometry control moves
for _c in (shape_sel, res, rTop, rBot, cyTop, cyBot, rHole,
           sxf, syf, lxf, lyf, radiate):
    _c.on_value_change(lambda _: draw_mask())
draw_mask()

ui.run(port=5500, title='RealSound', reload=True)


# ============================================================== MODIFYING ====
# Add a slider      -> knob('key', 'Label', lo, hi, default, step) in a section.
#                      Automatically saved, loaded and packed (knob registers it
#                      in CONTROLS). If it changes geometry, also add it to
#                      body_ir() AND solve_body() -- those signatures are the
#                      cache keys, so they must stay in sync.
# Add a dropdown    -> pick('key', [...], default, 'Label')  -- same deal.
# Live-preview it   -> append the control to the loop above draw_mask().
# Column widths     -> CARD (left) and VIEW (right).
# Centre / stack    -> the wrapping ui.row: 'justify-center' centres the pair;
#                      drop it to 'w-full' and the columns wrap on narrow windows.
# Plot colours      -> FG / BG constants, used by style_axes().
# More views        -> another ui.column() in the right card + a draw_*()
#                      function; call it from on_play().
#
# Note: rendering is synchronous, so the tab waits during a big pack export
# (each note is a Python-loop string synth, ~0.2 s). Fine for tens of notes.
