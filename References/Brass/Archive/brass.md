

Brass track · MD

\# Brass Track — Physical Modelling of Brass Instruments

&#x20;

> \*\*Companion to `PROJECT\_GUIDE.md`.\*\* This is the brass equivalent of the woodwind

> charter: it teaches the brass-specific physics, maps what we reuse against what we

> build, and lays out a gated milestone ladder. It assumes the woodwind toolkit

> (delay lines, fractional delay, loss filter, scattering junctions) is in place.

> Same working agreement as the parent guide — teach the physics before the code,

> respect the ladder, validate with the impulse test, and flag physics that is off.

&#x20;

\---

&#x20;

\## 0. Where This Sits

&#x20;

\*\*Revised workflow (this reorder):\*\* \*\*Brass → String → Percussion → Voice (deferred).\*\*

&#x20;

Brass was formerly a single milestone (parent guide §8, Milestone 6). It is now

promoted to a full track. The old voice-oriented milestones (glottal source, vocal

tract, realism, transitions, nasal coupling — parent §8 M7–M11) move \*after\* string

and percussion.

&#x20;

\*\*Why brass is a cheap next step.\*\* The brass track leans almost entirely on

infrastructure the woodwind work already produced:

&#x20;

| Woodwind milestone | What it gave us | Role in brass |

|--------------------|-----------------|---------------|

| M1 two-delay tube + fractional delay | The 1-D waveguide, correct tuning | The bore itself; the slide needs fractional delay badly |

| M2 frequency-dependent loss | Warm decay, HF loss per round trip | Same wall/thermal losses; unchanged |

| M3 bore sections / scattering junctions | Multi-section bore, cone vs cylinder | \*\*The bell is a cascade of junctions\*\* — M3 \*is\* the bell engine |

| M5 radiation + bell (parent) | Frequency-dependent radiation | Central to brass, not optional |

&#x20;

So brass is mostly: \*\*new source (lips), a seriously-modelled bell, a mouthpiece

resonator, and one genuinely new propagation physics (brassiness).\*\*

&#x20;

\---

&#x20;

\## 1. What Is Genuinely New in Brass

&#x20;

Everything else is reused. The new physics, shortest form:

&#x20;

1\. \*\*The source is a mechanical oscillator, not a pressure valve.\*\* The lips have

&#x20;  mass and stiffness and their own resonance. The player \*tunes that resonance\* to

&#x20;  pick a note. This is qualitatively different from the clarinet reed's hard clip.

2\. \*\*The bell tunes the instrument.\*\* A plain closed–open cylinder gives odd

&#x20;  harmonics on a non-harmonic grid — useless as a brass instrument. The flaring

&#x20;  bell (plus mouthpiece) \*shifts the resonances into a near-harmonic series\*. The

&#x20;  harmonicity is engineered by geometry, not intrinsic to the tube.

3\. \*\*Loud brass is nonlinear in the air itself.\*\* At high amplitude the wave steepens

&#x20;  as it travels the long bore, dumping energy into high harmonics — the bright,

&#x20;  cutting \*cuivré\*. A linear waveguide cannot produce this at any source setting.

4\. \*\*Pitch changes by changing tube length.\*\* Valves add discrete segments; the

&#x20;  trombone slide is continuous. (Woodwinds changed effective length by opening

&#x20;  toneholes — a different mechanism.)

\---

&#x20;

\## 2. The Physics of Brass Sound

&#x20;

\### 2.1 The lip reed is a driven oscillator, not a pressure-controlled valve

&#x20;

The clarinet reed (parent §2.6) is essentially a spring-loaded flap whose closure is

governed by the pressure across it — model it as a static nonlinearity (`clip`). The

brass player's lips are different: they are a \*\*damped mass–spring system with their

own natural frequency\*\*, driven by the pressure difference between the mouth and the

mouthpiece, and their motion modulates the flow into the instrument. The lips are an

active oscillator that the bore then \*entrains\*.

&#x20;

The minimal tractable model is a \*\*single-mass lip valve\*\*: a driven damped harmonic

oscillator whose displacement opens and closes the gap, feeding a flow that depends

on the gap area and the pressure drop (a Bernoulli-type relation). Three parameters

map to real playing:

&#x20;

\- \*\*Stiffness/tension → lip resonance frequency\*\* (the player's "embouchure"). This

&#x20; is the primary control that selects which note sounds.

\- \*\*Mass\*\* — larger for low brass; sets the range of achievable resonances.

\- \*\*Blowing pressure\*\* — the DC drive; sets amplitude and, past a threshold,

&#x20; pushes the system into and out of the \*cuivré\* regime (see 2.7).

\*\*Striking regime — a known subtlety, flagged honestly.\*\* Reeds are classified as

\*inward-striking\* (blown-closed, e.g. the clarinet — plays \*below\* the reed

resonance) or \*outward-striking\* (blown-open — plays \*above\*). Lips are genuinely

harder to pin down: the physical lip has both a longitudinal ("swelling") and a

transverse ("swinging door") motion, and the literature (Yoshikawa; Adachi \& Sato;

Cullen, Gilbert \& Campbell) converges on a \*\*transverse / sliding-door\*\* picture as

the best fit, with the sounding note typically sitting \*\*just below the lip's

mechanical resonance\*\*. For the lab, start with a single-mass outward-striking valve

tuned so the note locks just under the lip resonance; upgrade to a two-degree-of-

freedom lip model only if the attack transients or the high register feel wrong. Do

not claim the one-mass model is \*the\* truth — it is the tractable approximation.

&#x20;

\### 2.2 Harmonic selection and the natural series

&#x20;

Because the lip is an oscillator with a tunable resonance, the player selects a note

by moving the lip resonance near a bore resonance; the two \*\*lock\*\*, and that mode

sounds. Raising lip tension jumps to the next-higher mode. On a bugle (no valves,

fixed tube), this is the \*entire\* melodic mechanism — you play the natural harmonic

series (the "bugle calls") purely by embouchure. This is the same lock-in idea as the

parent guide's §2.6, made central here.

&#x20;

\### 2.3 The bore: mostly cylindrical, and why that alone is wrong

&#x20;

Trumpet and trombone bores are \*\*predominantly cylindrical\*\* in the middle, with a

tapered leadpipe at the mouthpiece end and a rapidly flaring bell at the far end.

Cornet, flugelhorn, French horn, and tuba are \*\*more conical\*\*.

&#x20;

This connects straight to woodwind M3. A cone (parent §2.5) restores the full

harmonic series on its own; a pure cylinder closed at one end gives \*\*odd harmonics

only, on a non-harmonic grid\*\* (`(2n−1)·c/4L`). So:

&#x20;

\- A predominantly \*\*conical\*\* brass (horn, tuba) gets much of its harmonicity from

&#x20; the taper — like the saxophone did.

\- A predominantly \*\*cylindrical\*\* brass (trumpet, trombone) would, without help, be

&#x20; a badly-tuned clarinet. The \*\*bell and mouthpiece do the harmonic-alignment work\*\*.

This is the key insight: \*a straight cylinder + lips is not a brass instrument.\* You

must add the bell before the resonances form a usable series. Expect the impulse test

of the bare cylinder-plus-lip stage (Milestone B1) to show odd harmonics on the wrong

grid — that "wrongness" is the motivation for B2.

&#x20;

\### 2.4 The bell: cutoff frequency, reflection/radiation split, harmonic alignment

&#x20;

The flaring bell is the single most important acoustic element in brass. Model it as

a cascade of short sections of rapidly increasing area (a \*\*Bessel horn\*\* profile) —

i.e. exactly the M3 scattering-junction machinery, with the junction reflection

coefficients shrinking toward zero as the bore widens (parent §2.7). The bell does

three things at once:

&#x20;

1\. \*\*Frequency-dependent reflection (the cutoff).\*\* There is a \*\*cutoff frequency\*\*

&#x20;  set by the flare rate. \*Below\* cutoff, long-wavelength waves cannot negotiate the

&#x20;  flare and reflect back into the tube — these are the frequencies that form the

&#x20;  standing-wave resonances. \*Above\* cutoff, the bell is effectively transparent and

&#x20;  radiates efficiently. So the bell is a \*\*low-pass reflector / high-pass radiator\*\*,

&#x20;  sharpening the parent guide's §2.7 statement.

2\. \*\*Harmonic alignment.\*\* The flare \*\*raises the frequencies of the low modes\*\*

&#x20;  (the wave "sees" an effectively shorter tube at low frequencies). Combined with the

&#x20;  mouthpiece's effect on the high modes (2.5), this shifts resonances 2, 3, 4, 5, 6…

&#x20;  so they land close to a harmonic series `2f, 3f, 4f…`. The instrument is

&#x20;  deliberately built so its resonance peaks \*fall on a harmonic series\*. This is why

&#x20;  brass plays in tune across a wide range from one tube.

3\. \*\*Directivity.\*\* The bell is a large aperture, so high frequencies \*\*beam

&#x20;  forward\*\*. Relevant for the app's spatialisation, not for getting the first sound.

\### 2.5 The mouthpiece: a Helmholtz resonator and the "brass formant"

&#x20;

The mouthpiece \*\*cup + backbore\*\* forms a \*\*Helmholtz-like resonance\*\* (the

"popping frequency," roughly a few hundred Hz to \~1 kHz depending on the instrument;

trumpet mouthpieces sit around the high-hundreds of Hz). This does two jobs:

&#x20;

\- It \*\*couples the lips to the air column\*\* and makes the upper register speak.

\- It \*\*boosts a band\*\* of the spectrum, producing a broad resonance emphasis often

&#x20; called the \*\*brass formant\*\* (a bright band typically in the low kHz). This is a

&#x20; large part of what makes brass sound like brass rather than a buzzing pipe.

Model it as a small lumped resonator (a Helmholtz element / short wide-then-narrow

section) inserted between the lip source and the bore. Its effect on the \*high\* modes

complements the bell's effect on the \*low\* modes — together they complete the

harmonic alignment of 2.4.

&#x20;

\### 2.6 The pedal tone / privileged frequency

&#x20;

A subtle but important point that \*will\* confuse impulse tests. The lowest playable

note (the \*\*pedal tone\*\*) is \*\*not a strong air-column resonance\*\*. If you impulse-

test a well-tuned brass bore you will see strong peaks at `2f, 3f, 4f…` but a weak or

absent peak at the fundamental `f`. Yet the pedal note plays. Why: the nonlinear

lip–bore feedback locks to a \*\*periodic oscillation whose harmonics coincide with the

existing upper resonances\*\*, so the ear hears the fundamental via harmonic spacing

(the missing-fundamental effect, parent §3.1) even though the tube barely resonates

there. \*\*Expectation to set:\*\* do not "fix" the model when the impulse test shows a

weak fundamental — that is physically correct. Validate the pedal by \*playing\* it

(coupled loop), not by the passive spectrum.

&#x20;

\### 2.7 Brassiness: nonlinear wave steepening and shock formation

&#x20;

This is the defining feature of loud brass and the one place the linear waveguide

\*\*fails on principle\*\*. At high blowing pressure the acoustic pressure in the bore is

large enough that propagation is \*\*weakly nonlinear\*\*: higher-pressure parts of the

wave travel slightly faster, so the waveform \*\*steepens as it travels down the long

tube\*\*, transferring energy progressively into high harmonics and approaching a

\*\*near-shock\*\* front. This is heard as the sound "blooming" from mellow to brilliant,

cutting, metallic (\*cuivré\*) as the player crescendos.

&#x20;

Consequences for modelling:

&#x20;

\- \*\*A linear delay-line bore cannot produce this at any source setting.\*\* The

&#x20; brightness is generated \*in the air column\*, not at the lips. This is a common

&#x20; omission and worth flagging loudly.

\- The standard fix is a \*\*weakly-nonlinear propagation term along the delay lines\*\*

&#x20; (a Burgers-equation-type amplitude-dependent steepening; see Menguy \& Gilbert,

&#x20; Thompson \& Strong, Msallam et al.). Practically: an amplitude-dependent

&#x20; high-frequency generation applied cumulatively along the bore, gated by level so

&#x20; quiet notes stay mellow and loud notes brighten.

\- It is \*\*amplitude- and length-dependent\*\*: longer instruments (trombone) and louder

&#x20; dynamics steepen more, which matches real trombone \*cuivré\* being so pronounced.

Treat this as its own milestone (B6). It is what separates "a convincing bugle" from

"a trumpet at fortissimo."

&#x20;

\### 2.8 Pitch change: valves and slides

&#x20;

Brass change pitch by \*\*changing the length of the air column\*\*, not by opening

toneholes.

&#x20;

\- \*\*Valves (trumpet, tuba, horn):\*\* each valve routes the air through an \*\*extra

&#x20; length of tubing\*\*, lowering pitch a fixed interval (2nd valve ≈ semitone, 1st ≈

&#x20; tone, 3rd ≈ minor third; combinations sum). Model each engaged valve as an

&#x20; \*\*added delay segment\*\* (with its own bends/junctions — the 1-D approximation is

&#x20; fine per the parent guide's bends discussion). \*\*Known intonation subtlety:\*\* valve

&#x20; \*combinations\* run sharp because the added lengths don't scale correctly for the

&#x20; now-longer instrument (the 1+3 and 1+2+3 fingerings especially) — real players

&#x20; compensate with slides. Worth reproducing rather than hiding.

\- \*\*Slide (trombone):\*\* a \*\*continuously variable\*\* length. This is where \*\*fractional

&#x20; delay earns its keep\*\* — a smooth glissando needs sub-sample-accurate, click-free

&#x20; length changes, and the length must be \*\*parameter-smoothed\*\* (one-pole ramp) or

&#x20; you get zipper noise (parent §5, §6.1). The trombone is arguably the \*\*cleanest

&#x20; first target\*\* because the slide is one continuous parameter and the bore is mostly

&#x20; cylindrical — no valve-junction bookkeeping.

\### 2.9 Mutes and directivity (stretch)

&#x20;

A \*\*mute\*\* inserted in the bell adds a \*\*coupled resonator\*\* at the radiation end that

reshapes the spectrum: a straight mute adds a nasal high-pass emphasis, a cup mute

darkens, a harmon/wah mute adds a strong movable resonance. Model as an extra

filter/resonator at the bell output. Combine with the frequency-dependent

\*\*directivity\*\* from 2.4 for a convincing spatial model in the app.

&#x20;

\---

&#x20;

\## 3. What We Reuse vs What We Build

&#x20;

\*\*Reused unchanged\*\* (from the woodwind track):

\- Two-delay-line bore with fractional delay (M1)

\- Frequency-dependent loss filter with carried state (M2)

\- Scattering-junction / multi-section bore engine (M3)

\- The coupled source↔resonator feedback loop \*structure\* (parent §9)

\*\*New for brass:\*\*

\- The \*\*lip oscillator\*\* source (driven damped mass–spring valve; replaces the reed

&#x20; `clip`)

\- The \*\*bell\*\* as a tuned Bessel-horn cascade (specialisation of M3 + radiation)

\- The \*\*mouthpiece\*\* Helmholtz resonator

\- \*\*Nonlinear propagation\*\* (brassiness) along the bore

\- \*\*Length control\*\*: discrete valve segments and/or a continuous smoothed slide

\---

&#x20;

\## 4. The Brass Milestone Ladder

&#x20;

Each is a self-contained experiment, proven in the Python lab, validated by impulse

test (or by playing, where the impulse test is misleading — see B4/pedal). Update

status as work lands.

&#x20;

| # | Milestone | Goal | Key physics | Status |

|---|-----------|------|-------------|--------|

| B0 | \*\*Foundation check\*\* | Confirm M1–M3 toolkit (delay + fractional delay + loss + junctions) is reusable as-is for a closed–open bore. | Reuse audit | ☐ |

| B1 | \*\*Lip oscillator on a bare cylinder\*\* | Self-sustaining buzz; harmonic selection by lip tension (bugle behaviour) on a plain closed–open tube. Expect odd harmonics on a \*non-harmonic\* grid — this is the motivation for B2. | Lip valve as driven oscillator; lock-in (2.1–2.2) | ☐ |

| B2 | \*\*The bell (Bessel-horn flare)\*\* | Add the flare; impulse-test shows low modes shift and 2–6 line up into a near-harmonic series; demonstrate the cutoff (low reflects, high radiates). Turns the buzzing pipe into a bugle. | Cutoff, reflection/radiation split, harmonic alignment (2.4) | ☐ |

| B3 | \*\*The mouthpiece\*\* | Add cup+backbore Helmholtz resonator; observe the response boost / brass formant and easier upper-register speech. | Helmholtz resonance, brass formant (2.5) | ☐ |

| B4 | \*\*Full playable natural instrument\*\* | Assemble lip + mouthpiece + cylindrical body + bell; play the full natural series incl. the pedal tone by embouchure alone. Validate pedal by \*playing\*, not by the passive spectrum. | Coupled loop; privileged/pedal frequency (2.6) | ☐ |

| B5 | \*\*Pitch mechanism\*\* | Add length control: trombone slide (continuous, smoothed, fractional-delay glissando) and/or trumpet valves (discrete added segments; reproduce the combination-sharpness intonation error). | Variable bore length; smoothing (2.8, parent §6.1) | ☐ |

| B6 | \*\*Brassiness\*\* | Add amplitude-dependent nonlinear propagation; A/B the \*cuivré\* bloom vs the linear model across a crescendo. This is what makes it sound like real brass at forte. | Weakly-nonlinear wave steepening / shock formation (2.7) | ☐ |

| B7 | \*\*Mutes + directivity\*\* \*(stretch)\* | Bell-mounted resonators (straight/cup/harmon) and frequency-dependent forward beaming. | Coupled bell resonator; radiation directivity (2.9) | ☐ |

&#x20;

\*\*Validation gate.\*\* Impulse-test the resonator and confirm peaks land where theory

predicts \*before\* trusting the source — \*\*except\*\* at the pedal fundamental, where a

weak/absent peak is correct (2.6). "In tune but harmonics on the wrong grid" → the

bell/mouthpiece tuning isn't done (B2/B3). "Won't brighten when blown hard" → the

nonlinear propagation is missing or ungated (B6). "Zipper on the slide" → smooth the

length parameter (B5). "Blows up" → a loop gain > 1, same as always (parent §6.3).

&#x20;

\---

&#x20;

\## 5. Known Subtleties to Watch

&#x20;

Tracked here so they get flagged in future sessions (parent §1, §7 style).

&#x20;

1\. \*\*A cylinder + lips is not a brass instrument.\*\* The harmonic series is engineered

&#x20;  by the bell and mouthpiece, not intrinsic to the tube. Don't expect a usable

&#x20;  series before B2/B3.

2\. \*\*The pedal fundamental is weak or absent in the passive spectrum\*\* and that is

&#x20;  correct. Validate it by playing, via the missing-fundamental mechanism (2.6).

3\. \*\*Linear waveguides cannot produce brassiness.\*\* The brightness of loud brass is

&#x20;  generated in the air column by nonlinear steepening, not at the lips (2.7).

4\. \*\*Lip striking regime is genuinely debated.\*\* The one-mass model is a tractable

&#x20;  approximation, not ground truth; a transverse/two-DOF lip is more faithful (2.1).

5\. \*\*Valve combinations play sharp\*\* by physics, not by mistake (2.8) — reproduce it.

6\. \*\*The slide demands fractional delay and parameter smoothing\*\* together, or

&#x20;  glissandi click and zip (2.8).

\---

&#x20;

\## 6. First Concrete Target

&#x20;

Recommend starting the lab work on the \*\*trombone\*\*: predominantly cylindrical bore

(so B1–B4 are the clean cylinder-plus-bell case), and pitch is one continuous slide

parameter (exercising fractional delay directly, no valve bookkeeping). Its long tube

also makes the B6 \*cuivré\* effect dramatic and easy to hear in an A/B. Move to the

\*\*trumpet\*\* for the valve mechanism and combination-intonation modelling, and treat

the \*\*French horn / tuba\*\* (more conical, horn's hand-in-bell technique) as the

higher-fidelity follow-ups once the cylindrical case is solid.

&#x20;

\---

&#x20;

\*Living document. Update §4 status as milestones land; record any new physics

corrections here and in the parent guide's §7.\*

