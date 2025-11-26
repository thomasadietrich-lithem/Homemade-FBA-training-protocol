# 🧠 FBA RDK Training — PsychoPy Implementation

A Python/PsychoPy reconstruction of the University of Rochester’s Feature-Based Attention (FBA) visual motion training task.

---

## 📌 Overview

This repository contains a working PsychoPy implementation of an FBA Random-Dot Kinematogram (RDK) training task used in visual rehabilitation protocols from the Huxlin Lab (University of Rochester).

It was recreated by reverse-engineering Matlab + Psychtoolbox code made publicly available in their GitHub repository, with the goal of preserving the logic of:

- stimulus geometry  
- staircase dynamics  
- dot motion parameters  
- timing  
- eccentricity mapping  
- response rules  
- auditory feedback  

This version is intended for **personal training**, with a simplified setup that does not depend on Matlab, Psychtoolbox, or cloud infrastructure.

---

## 🧩 Files in This Repository

```text
cb_fba_training_psychopy.py        # Main training script – "global tilt" version
cb_fba_training_psychopy_DR.py     # Alternative version – "Direction Range enabled"
analyse_fba_progress.py            # Script to visualize progress over sessions
Two things are created automatically when you run the program:

📁 1. /data/ folder (auto-created)
Stores:

trial-level .csv files

session-level .json summaries

📁 2. monitor_profiles.json (auto-created)
Stores per-monitor screen calibration so that stimulus geometry and eccentricity remain correct when switching displays.

You do not need to create these manually.

🔧 Requirements
Python 3.10 → 3.12

PsychoPy 2024.2+

numpy

Standard Python libs: json, csv, time, os

Download PsychoPy:
https://www.psychopy.org/download.html

🎯 Scientific Fidelity — What Has Been Reproduced
The following elements match the logic and parameter ranges found in the Rochester Matlab code and related publications:

✔ Stimulus parameters
RDK with motion direction discrimination around a main axis

Dot density: 3.5 dots/deg²

Dot size: 14 arcmin (converted to pixels based on calibration)

Dot lifetime: 200 ms

Dot speed: 10 deg/s

Circular aperture: 2.5° radius

Stimulus duration: 500 ms

Background: mid-gray

✔ Staircase system
3 interleaved staircases

Angle ranges identical to the Matlab set:
angle_range = [85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]

Staircase rule (per staircase):

+1 level after 3 consecutive correct responses

−1 level after a single incorrect response

✔ Fixation and pre-cue
Dual-ring fixation dot (black outer ring + white inner dot)

Pre-cue: a line from fixation to the stimulus location

✔ Response rules
angle_set = 0 → training on a horizontal axis
→ respond UP / DOWN depending on whether the global motion is tilted upward or downward

angle_set = 1 → training on a vertical axis
→ respond LEFT / RIGHT depending on whether the global motion is tilted leftward or rightward

✔ Auditory feedback
1000 Hz = trial start

1200 Hz = correct response

800 Hz = incorrect response

🖥️ Monitor Calibration (Auto-Saved)
Because this program may be used on different displays, a simple calibration system is included to keep geometry consistent.

On first use with a new monitor, the script asks for:

physical screen width (in centimeters)

viewing distance (in centimeters)

it also reads the current screen resolution in pixels

From these values, it computes a conversion between:

pixels ↔ degrees of visual angle

pixels ↔ arcminutes

using the same geometry as the original Matlab/Psychtoolbox code:

text
Copier le code
theta = atan( (screen_width_cm / 2) / viewing_distance_cm )   # in radians
theta_deg = theta * 180 / pi
arcmin_per_pixel = theta_deg * 60 / (screen_width_in_pixels / 2)

deg_to_pix = (deg * 60) / arcmin_per_pixel
This ensures that:

stimulus eccentricity (H, V in degrees)

dot size in arcmin

aperture radius in degrees

are consistent across sessions and monitors.

Calibration is then saved into:

text
Copier le code
monitor_profiles.json
using a key like "OS:widthxheight" (e.g. "Windows:1920x1080"), and automatically reused the next time the script is run on that monitor.

🔀 Two Variants of the Task (for Rochester’s Feedback)
This repository currently provides two closely related variants of the training task, so that the Rochester team can confirm which one best matches their current clinical implementation.

1️⃣ cb_fba_training_psychopy.py — "Global Tilt" Version
This version reproduces exactly the behavior observed in the Matlab file we had access to.

In that Matlab code, the per-dot direction was computed using a line of the form:

matlab
Copier le code
vectors = pi * (angle + normrnd(0, 0, ndots, 1)) / 180;
Because the standard deviation of the Gaussian is set to 0 (normrnd(0, 0, ...)), all dots share exactly the same direction.
The task therefore becomes a small global-tilt discrimination around the main axis (e.g. slightly above vs. slightly below the horizontal).

In other words:

the staircase adjusts the tilt size around the axis

but there is effectively no direction-range dispersion across dots in this particular Matlab file.

This "global tilt" version is implemented in:

text
Copier le code
cb_fba_training_psychopy.py
and is faithful to that specific Matlab behavior.

2️⃣ cb_fba_training_psychopy_DR.py — "Direction Range Enabled" Version
This alternative version implements a true direction-range stimulus, closer to what is described in the Huxlin/Tadin publications and patent.

Here:

a central direction angle_deg is defined (corresponding to the main axis + tilt),

and each dot’s direction is drawn from a Gaussian distribution around this central direction:

python
Copier le code
noise_deg = np.random.normal(loc=0.0, scale=angle_deviationP, size=n_dots)
dot_angles_deg = angle_deg + noise_deg
when a dot’s lifetime expires and it is respawned, it is given a new random direction sampled from the same distribution.

Thus, the direction range is controlled by the staircase variable angle_deviationP:

large values (e.g. 85°) → wide dispersion of motion directions

small values (e.g. 2°, 1°, 0.5°) → very subtle tilt around the axis

This version is implemented in:

text
Copier le code
cb_fba_training_psychopy_DR.py
and the session summary files are tagged with:

json
Copier le code
"task": "FBA_RDK_DirectionRange"
📊 Analytics
analyse_fba_progress.py generates progress graphs based on the .json summaries in /data.

It displays:

direction threshold progression over sessions (primary clinical metric)

accuracy (%) across sessions

Usage:

bash
Copier le code
python analyse_fba_progress.py
The script currently loads all *_FBA_*_summary.json files, so it will include both “global tilt” and “Direction Range” sessions unless further filtered.

▶️ Running the Training
You can run either variant:

bash
Copier le code
python cb_fba_training_psychopy.py        # Global tilt version
python cb_fba_training_psychopy_DR.py     # Direction Range enabled version
In both cases, you will be prompted for:

angle set (0 = horizontal axis / UP–DOWN, 1 = vertical axis / LEFT–RIGHT)

stimulus eccentricity in degrees (H, V, visual field convention)

subject ID

Then the training session begins.

❗ Features Not Included (by Design)
The original Matlab code (full clinical tool) contains:

an automatic displacement mechanism to move the stimulus deeper into the blind field after stable performance,

cloud/backend logic (AWS S3) for remote data upload.

These have not been implemented here:

stimulus location is chosen manually before each session,

all data are stored locally.

📬 Acknowledgement
This work is a Python/PsychoPy reinterpretation of the structure of an FBA training tool shared by the Huxlin Lab (GitHub: https://github.com/huxlinlab).

cb_fba_training_psychopy.py closely reproduces the behavior of the specific Matlab file we could inspect (with direction-range effectively disabled by normrnd(0, 0, ...)).

cb_fba_training_psychopy_DR.py re-enables a true direction-range stimulus, inspired by their published descriptions.

Feedback from the Rochester team is warmly welcomed, in particular regarding:

which of these two variants best matches their current clinical implementation,

and whether the calibration approach is sufficient for their standards.

Contact:
Thomas Dietrich — thomas.a.dietrich@gmail.com
