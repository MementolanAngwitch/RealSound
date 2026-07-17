"""
tuning_test.py — verify the fractional-delay string is in tune.

Renders each note of a scale on an integer-delay and a fractional-delay
Karplus-Strong string, measures the ACTUAL pitch of the rendered audio,
and reports the error in cents against the target.

Why measure the audio (not the parameters): the pitch is an emergent
property of the feedback loop. Printing the target f or the integer N tells
you nothing about what actually came out. Two measurement gotchas handled
below: (1) the triangular pluck is all-positive, so it carries a DC offset
that fools a raw FFT peak-picker -> remove the mean first; (2) integer FFT
bins are too coarse for cents-level accuracy -> autocorrelate and
parabolically interpolate the peak lag for sub-sample period resolution.
"""

import numpy as np

fs = 44100


def ks(beta=0.5, f=220, duration=1.0, rho=0.999, fractional=True):
    """Karplus-Strong string on an explicit delay line.

    fractional=True  -> loop delay = N + frac + 0.5 = fs/f exactly (in tune)
    fractional=False -> loop delay = N + 0.5 (integer-quantized pitch)
    The interpolation is INSIDE the feedback path (buf[w] uses the
    interpolated y), which is what actually retunes the string.
    """
    D = fs / f - 0.5                 # delay the loop must realize (0.5 = avg filter)
    N = int(D)                       # integer part -> buffer taps
    frac = (D - N) if fractional else 0.0

    p = int(beta * N)
    pluck = np.concatenate((np.linspace(0, 1, p), np.linspace(1, 0, N - p)))

    L = N + 2                        # buffer longer than the deepest tap (N+1)
    buf = np.zeros(L)
    buf[:N] = pluck
    out = np.zeros(int(fs * duration))
    w = 0
    last = 0.0
    for i in range(len(out)):
        r0 = (w - N) % L             # tap at delay N
        r1 = (w - N - 1) % L         # tap at delay N+1 (one sample older)
        y = (1 - frac) * buf[r0] + frac * buf[r1]   # fractional read; older tap gets frac
        buf[w] = 0.5 * (y + last) * rho             # KS loss filter, fed the interpolated y
        last = y
        out[i] = y
        w = (w + 1) % L
    return out


def measure_period(sig, skip=2000):
    """Estimate fundamental (Hz) of a rendered note to sub-sample accuracy."""
    s = sig[skip:]                   # skip the attack transient
    s = s - np.mean(s)               # remove DC (the pluck's positive offset)
    ac = np.correlate(s, s, "full")[len(s) - 1:]   # autocorrelation, positive lags
    lo = 30                          # ignore tiny lags (avoid the zero-lag peak)
    k = np.argmax(ac[lo:]) + lo      # coarse period in samples
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c)            # parabolic sub-sample refinement
    return fs / (k + d)


def cents(f_measured, f_target):
    return 1200 * np.log2(f_measured / f_target)


def make_scale(root_hz, semitones):
    return root_hz * 2 ** (np.array(semitones) / 12)


if __name__ == "__main__":
    scale = make_scale(220, [0, 2, 4, 5, 7, 9, 11, 12])   # A major
    print(f"{'target':>8} | {'integer':>8} {'cents':>6} | {'frac':>8} {'cents':>6}")
    for ft in scale:
        fi = measure_period(ks(f=ft, fractional=False))
        fm = measure_period(ks(f=ft, fractional=True))
        print(f"{ft:8.2f} | {fi:8.2f} {cents(fi, ft):6.1f} | {fm:8.2f} {cents(fm, ft):6.1f}")
