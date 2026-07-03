"""
render_assets.py — the I/O layer.

The analysis/synthesis engines return raw numpy arrays and never touch disk.
This script turns them into deliverable assets: normalised WAV files and
analysis PNG figures. Run once to (re)generate everything the walkthrough
notebook references. Mirrors brass's render_assets.py role.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile

import string_analysis as A
import string_synth as S
from string_analysis import FS

OUT = "/mnt/user-data/outputs"
AUD = os.path.join(OUT, "audio")
FIG = os.path.join(OUT, "figures")
os.makedirs(AUD, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
np.random.seed(0)

def save_wav(name, sig, gain=0.9):
    x = sig / (np.max(np.abs(sig)) + 1e-9) * gain
    wavfile.write(f"{AUD}/{name}.wav", int(FS), (x * 32767).astype(np.int16))
    print(f"  wav  {name}.wav")

def save_fig(name):
    plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=110)
    plt.close(); print(f"  fig  {name}.png")

# --- S0 --------------------------------------------------------------------
print("S0 Karplus–Strong")
save_wav("s0_karplus", S.karplus_strong(220.0))

# --- S1 --------------------------------------------------------------------
print("S1 fixed–fixed pluck")
y_s1 = S.pluck_string(220.0, pluck_frac=0.20, loss_scalar=0.998)
save_wav("s1_pluck_L5", y_s1)
plt.figure(figsize=(9, 3)); f, db = A.spectrum_db(y_s1)
plt.plot(f, db); plt.xlim(0, 12 * 220); plt.ylim(-90, 3)
for n in range(1, 13):
    plt.axvline(n * 220, color='k', lw=0.4, alpha=0.4)
plt.title("S1: all harmonics; nulls at n=5,10 (pluck at L/5)")
plt.xlabel("Hz"); plt.ylabel("dB"); save_fig("s1_comb")

# --- S2 --------------------------------------------------------------------
print("S2 frequency-dependent loss")
save_wav("s2_warm", S.pluck_string(220.0, fc=5500, gain=0.999, dur_s=2.0))

# --- S3 --------------------------------------------------------------------
print("S3 stiffness / dispersion")
y_s3 = S.pluck_string(110.0, pluck_frac=0.15, fc=4500, gain=0.9990,
                      allpass=(-0.6, 40), dur_s=2.5)
save_wav("s3_stiff", y_s3)
f1, B, ns, fns = A.fit_inharmonicity(y_s3, 110.0)
plt.figure(figsize=(9, 3))
plt.plot(ns, fns / (ns * f1), 'o-')
plt.axhline(1.0, color='k', lw=0.5)
plt.title(f"S3: partial stretch (fitted B = {B:.2e})")
plt.xlabel("harmonic n"); plt.ylabel("f_n / (n·f1)"); save_fig("s3_stretch")

# --- S4 --------------------------------------------------------------------
print("S4 bowed string")
y_s4, stick = S.bow_string(196.0, F_b=1.20, return_stick=True)
save_wav("s4_bow", y_s4)
fm = A.estimate_f0(y_s4[int(0.5 * FS):]); per = int(FS / fm); t0 = int(1.0 * FS)
plt.figure(figsize=(9, 3)); plt.plot(y_s4[t0:t0 + 3 * per])
plt.title("S4: Helmholtz sawtooth at the bridge (emergent)")
plt.xlabel("samples"); save_fig("s4_helmholtz")

# --- S5 --------------------------------------------------------------------
print("S5 bridge + body (commuted)")
body = S.make_body_ir()
exc = S.pluck_excitation()
y_plain = S.string_loop(exc, 196.0)
y_body = S.string_loop(A.fftconvolve(exc, body), 196.0)
save_wav("s5_string_only", y_plain)
save_wav("s5_with_body", y_body)
plt.figure(figsize=(9, 3))
fa, da = A.spectrum_db(y_plain); fb, dbb = A.spectrum_db(y_body)
plt.plot(fa, da, alpha=0.6, label="string only")
plt.plot(fb, dbb, alpha=0.8, label="commuted body")
plt.xlim(0, 2000); plt.ylim(-80, 3); plt.legend()
plt.title("S5: body IR commuted into the excitation")
plt.xlabel("Hz"); plt.ylabel("dB"); save_fig("s5_body")

# --- S6 --------------------------------------------------------------------
print("S6 polarizations + unisons")
y_pol = S.two_polarizations(220.0)
y_uni = S.coupled_pair(220.0, 220.0 * 1.0015, 0.9975, 0.9975, eps=0.02)
save_wav("s6_polarizations", y_pol)
save_wav("s6_unisons", y_uni)
plt.figure(figsize=(9, 3))
plt.plot(np.arange(len(y_pol)) / FS, A.envelope_db(y_pol), label="two polarizations")
plt.plot(np.arange(len(y_uni)) / FS, A.envelope_db(y_uni), alpha=0.7, label="coupled unisons")
plt.ylim(-80, 5); plt.legend(); plt.xlabel("s"); plt.ylabel("dB")
plt.title("S6: two-stage decay + unison beating"); save_fig("s6_decay")

# --- S7 --------------------------------------------------------------------
print("S7 piano hammer")
y_soft, c_soft = S.hammer_strike(v0=0.002)
y_hard, c_hard = S.hammer_strike(v0=0.012)
save_wav("s7_pianissimo", y_soft)
save_wav("s7_fortissimo", y_hard)
plt.figure(figsize=(9, 3))
fs_, ds_ = A.spectrum_db(y_soft / np.max(np.abs(y_soft)))
fh_, dh_ = A.spectrum_db(y_hard / np.max(np.abs(y_hard)))
plt.plot(fs_, ds_, alpha=0.6, label="pianissimo")
plt.plot(fh_, dh_, alpha=0.8, label="fortissimo")
plt.xlim(0, 4000); plt.ylim(-80, 3); plt.legend()
plt.title("S7: harder strike → more high-partial energy")
plt.xlabel("Hz"); plt.ylabel("dB"); save_fig("s7_brightness")

print("\nassets rendered to", OUT)
