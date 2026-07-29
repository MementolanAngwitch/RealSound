
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

#  1. SETTINGS 
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


#  2. ENGINE GLUE
def current_mask(shape, res, rTop, rBot, cyTop, cyBot, rHole):
    #The body outline only -- cheap, no eigenproblem. Used by the preview.
    if shape == 'Square':
        return np.ones((res, res), dtype=bool)
    return rs.boolean_8_grid(res=res, rTop=rTop, rBot=rBot,
                             cyTop=cyTop, cyBot=cyBot, rHole=rHole)


@lru_cache(maxsize=16)
def solve_body(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole):
    #Body geometry -> modes. THE slow step, cached on exactly what it uses.
    if shape == 'Square':
        _K, evals, evecs = rs.orthotropic_plate(ax=ax, ay=ay, resolution=res)
        return evals, evecs, np.arange(res * res).reshape(res, res)
    mask = current_mask(shape, res, rTop, rBot, cyTop, cyBot, rHole)
    return rs.plate_from_mask(mask, ax=ax, ay=ay)


def snap_inside(mask, fx, fy):
    """Fractional (0-1) point -> a cell really inside the body. build_body_IR
    does idx[sx,sy]; an exterior cell holds -1, which would silently read the
    LAST mode row instead of raising."""
    res = mask.shape[0]
    i = min(max(int(round(fx * (res - 1))), 0), res - 1)
    j = min(max(int(round(fy * (res - 1))), 0), res - 1)
    if mask[i, j]:
        return i, j
    ii, jj = np.where(mask)
    k = int(np.argmin((ii - i) ** 2 + (jj - j) ** 2))
    return int(ii[k]), int(jj[k])


def assemble_ir(evals, evecs, idx, t, n_modes, Q, r_s, r_l, f_top, g_low,
                radiated, **osc):
    """
    POINT listen   : amp_k = phi_k(strike) * phi_k(listen)   -- a microphone at
                     one spot; a point on a node line hears nothing.
    RADIATED listen: amp_k = phi_k(strike) * mean_j phi_k(j) -- the net volume
                     of air the mode displaces. This is the compact-source
                     (monopole) approximation: valid while the body is small
                     compared to the wavelength, i.e. at low frequency.
                     Antisymmetric modes have +/- halves that cancel, so their
                     mean is ~0 and they go silent by themselves -- the dipole
                     cancellation a real body has, for free.

    Verified to reproduce rs.build_body_IR exactly when radiated=False.
    The three coupled low modes already carry their own radiation term
    (drive = top, radiate = top + air), so they are unchanged either way.
    """
    plate_f = f_top * np.sqrt(evals) / np.sqrt(evals[0])
    listen_vec = evecs.mean(axis=0) if radiated else evecs[r_l]   # <- the switch
    freqs_p, amps_p, Qs_p = [], [], []
    for k in range(1, n_modes):
        freqs_p.append(plate_f[k])
        amps_p.append(evecs[r_s, k] * listen_vec[k])
        Qs_p.append(Q)

    freqs3, ev3 = rs.three_oscillator_model(f_top=f_top, **osc)
    amps3 = [ev3[0, k] * (ev3[0, k] + ev3[2, k]) for k in range(3)]
    Qs3 = [30, 30, 15]

    freqs = np.concatenate([freqs3, freqs_p])
    amps = np.concatenate([g_low * np.array(amps3), amps_p])
    Qs = np.concatenate([Qs3, Qs_p])
    y = rs.modal_bank(freqs, amps, Qs, t)
    return y / np.max(np.abs(y))


@lru_cache(maxsize=16)
def body_ir(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole,
            n_modes, Q, f_top, g_low, sxf, syf, lxf, lyf,
            f_back, volume, hole_radius, m_top, m_back,
            radiated=False, ir_sec=0.6):
    """Modes -> impulse response. modal_bank and three_oscillator_model run
    inside assemble_ir (cavity params ride along via **osc)."""
    evals, evecs, idx = solve_body(shape, res, ax, ay, rTop, rBot, cyTop, cyBot, rHole)
    sx, sy = snap_inside(idx >= 0, sxf, syf)
    lx, ly = snap_inside(idx >= 0, lxf, lyf)
    t = np.arange(int(FS * ir_sec)) / FS
    return assemble_ir(evals, evecs, idx, t, n_modes, Q,
                       idx[sx, sy], idx[lx, ly], f_top, g_low, radiated,
                       f_back=f_back, Volume=volume, hole_radius=hole_radius,
                       m_top=m_top, m_back=m_back)


def make_voice(ir, beta, rho, duration):
    #f -> audio. This closure is the 'instrument' that play_sequence plays.
    def voice(f):
        s = rs.pluck_string(f=f, beta=beta, duration=duration, fs=FS, rho=rho)
        n = scipy.signal.fftconvolve(s, ir)
        n[-2000:] *= np.linspace(1, 0, 2000)
        return n
    return voice


def norm16(y):
    return np.int16(0.98 * y / np.max(np.abs(y)) * 32767)


#3. VIEWS 
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
    #Waveform (first 30 ms) and spectrum in dB -- verify by eye.
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


# 4. CONTROLS 
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

    with ui.card().classes(CARD):                       #left
        ui.label('RealSound').classes(f'{INK} text-lg')

        with ui.row().classes('w-full items-center no-wrap'):
            note_sel = pick('root', list(NOTES), 'E2', 'Root')
            mode_sel = pick('mode', ['Note', 'Chord', 'Scale'], 'Note', 'Play')

        with ui.expansion('String', value=True).classes(SEC):
            beta = knob('beta', 'Pluck position', 0.02, 0.50, 0.13, 0.01)
            rho = knob('rho', 'Sustain', 0.90, 0.999, 0.99, 0.001, '{:.3f}')
            dur = knob('duration', 'Note length (s)', 0.5, 4.0, 2.0, 0.1, '{:.1f}')

        with ui.expansion('Body shape').classes(SEC):
            shape_sel = pick('shape', ['Figure-8', 'Square'], 'Figure-8', 'Outline')
            res = knob('res', 'Grid resolution', 16, 44, 32, 2, '{:.0f}')
            rTop = knob('rTop', 'Upper bout radius', 0.08, 0.30, 0.18, 0.01)
            rBot = knob('rBot', 'Lower bout radius', 0.10, 0.35, 0.25, 0.01)
            cyTop = knob('cyTop', 'Upper bout centre', 0.50, 0.90, 0.68, 0.01)
            cyBot = knob('cyBot', 'Lower bout centre', 0.10, 0.50, 0.35, 0.01)
            rHole = knob('rHole', 'Soundhole radius', 0.00, 0.12, 0.05, 0.01)
            for k in (rTop, rBot, cyTop, cyBot, rHole):
                k.bind_visibility_from(shape_sel, 'value', lambda v: v == 'Figure-8')

        with ui.expansion('Plate').classes(SEC):
            ax = knob('ax', 'Stiffness along grain', 0.5, 8.0, 4.0, 0.1, '{:.1f}')
            ay = knob('ay', 'Stiffness across grain', 0.5, 8.0, 1.0, 0.1, '{:.1f}')
            f_top = knob('f_top', 'Top resonance (Hz)', 80, 400, 180, 5, '{:.0f}')
            Q = knob('Q', 'Resonance Q', 3, 60, 20, 1, '{:.0f}')
            n_modes = knob('n_modes', 'Number of modes', 4, 60, 30, 1, '{:.0f}')
            g_low = knob('g_low', 'Low-end boom', 0.0, 1.0, 0.1, 0.05)

        with ui.expansion('Strike and listen').classes(SEC):
            sxf = knob('sx', 'Strike x', 0.0, 1.0, 0.35, 0.05)
            syf = knob('sy', 'Strike y', 0.0, 1.0, 0.35, 0.05)
            radiate = flag('radiated', 'Radiated (whole surface)')
            lxf = knob('lx', 'Listen x', 0.0, 1.0, 0.60, 0.05)
            lyf = knob('ly', 'Listen y', 0.0, 1.0, 0.60, 0.05)
            for k in (lxf, lyf):                      # no single point when radiating
                k.bind_visibility_from(radiate, 'value', lambda v: not v)

        with ui.expansion('Air cavity').classes(SEC):
            f_back = knob('f_back', 'Back resonance (Hz)', 120, 320, 200, 5, '{:.0f}')
            volume = knob('volume', 'Cavity volume (m³)', 0.004, 0.025, 0.012, 0.001, '{:.3f}')
            hole_r = knob('hole_radius', 'Soundhole radius (m)', 0.02, 0.06, 0.041, 0.001, '{:.3f}')
            m_top = knob('m_top', 'Top plate mass', 0.05, 0.40, 0.15, 0.01)
            m_back = knob('m_back', 'Back plate mass', 0.02, 0.20, 0.05, 0.01)

        with ui.expansion('Sequence').classes(SEC):
            patt_sel = pick('pattern', list(PATTERNS), 'Major triad', 'Pattern')
            dt = knob('dt', 'Note spacing (s)', 0.0, 1.0, 0.35, 0.05)
        for w in (patt_sel, dt):
            w.bind_visibility_from(mode_sel, 'value', lambda v: v != 'Note')

    with ui.card().classes(VIEW):                        #  right 
        ui.label('Body').classes(f'{INK} text-lg')
        mask_box = ui.column().classes('w-full items-center')

        play = ui.button('Play').props('color=primary').classes('w-full')
        tuning = ui.label().classes(f'{MUTED} text-xs')
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


# 5. PRESETS 
def collect():
    #Every control's value -> a plain dict. This IS the instrument.
    return {k: el.value for k, el in CONTROLS.items()}


def apply(data):
    #Set every control from a dict; unknown keys ignored, missing keys kept.
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


# 6. ACTION 
def current_ir():
    return body_ir(shape_sel.value, int(res.value), ax.value, ay.value,
                   rTop.value, rBot.value, cyTop.value, cyBot.value, rHole.value,
                   int(n_modes.value), Q.value, f_top.value, g_low.value,
                   sxf.value, syf.value, lxf.value, lyf.value,
                   f_back.value, volume.value, hole_r.value,
                   m_top.value, m_back.value, radiate.value)


def render_current():
    #Whatever Play would produce, as float audio.
    voice = make_voice(current_ir(), beta.value, rho.value, dur.value)
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
    ui.notify(f'Rendering {len(chosen)} notes…')
    voice = make_voice(current_ir(), beta.value, rho.value, dur.value)
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


