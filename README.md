🧠 FBA RDK Training — PsychoPy Implementation

A Python/PsychoPy reconstruction of the University of Rochester’s Feature-Based Attention (FBA) direction-range visual training task.

📌 Overview

This repository contains a working PsychoPy implementation of the FBA direction-range Random-Dot Kinematogram (RDK) training task used in visual rehabilitation protocols from the Huxlin Lab (University of Rochester).
It was recreated by reverse-engineering the Matlab + Psychtoolbox code publicly available in their GitHub repository, with the goal of preserving the exact logic of:

> stimulus geometry
> staircase dynamics
> dot motion parameters
> timing
> eccentricity mapping
> response rules
> feedback sounds

This version is intended for personal training, with a simplified setup that does not depend on Matlab, Psychtoolbox, or cloud infrastructure.

🧩 Files in This Repository
cb_fba_training_psychopy.py     # Main training script (FBA RDK protocol)
analyse_fba_progress.py         # Script to visualize progress over sessions


Two additional things are created automatically when you run the program:

📁 1. /data folder (auto-created)
Stores:

> trial-level .csv files
> session-level .json summaries

📁 2. /monitor_profiles.json (auto-created)

> A file that stores screen calibration so that stimulus geometry remains correct across monitors.

You do not need to create these manually.

🔧 Requirements
> Python 3.10 → 3.12
> Dependencies
> PsychoPy 2024.2+
> numpy (standard)
> json, csv, time, os (standard)

Download PsychoPy:
https://www.psychopy.org/download.html

🎯 Scientific Fidelity — What Has Been Reproduced

The following elements match the logic found in the Rochester Matlab version:

✔ Stimulus parameters
> RDK with direction-range discrimination
> Dot density: 3.5 dots/deg²
> Dot size: 14 arcmin (converted to pixels based on calibration)
> Dot lifetime: 200 ms
> Dot speed: 10 deg/s
> Aperture: 2.5° radius, circular
> Stimulus duration: 500 ms
> Background: mid-gray

✔ Staircase system
> 3 interleaved staircases
> Angle ranges identical to the Matlab set (85 → 0.5°)
> +1 difficulty after 3 correct responses
> −1 after one incorrect

✔ Fixation and pre-cue
> Dual-ring fixation dot
> Pre-cue indicating expected axis (horizontal or vertical)

✔ Response rules
> angle_set = 0 → respond UP / DOWN
> angle_set = 1 → respond LEFT / RIGHT

✔ Auditory feedback
> 1000 Hz = trial start
> 1200 Hz = correct response
> 800 Hz = incorrect response

🖥️ Monitor Calibration (Auto-Saved)
> The script detects if the monitor has been used before. If not, it asks for:
> screen physical width (cm)
> viewing distance (cm)
> screen resolution
> It then computes a degree and arcmin → pixel conversion identical to the one used in the Matlab/Psychtoolbox version.
> This ensures that eccentricity (H, V in degrees) is geometrically accurate, even when switching monitors.

Calibration is saved into:
> monitor_profiles.json and reused automatically.

📊 Analytics
analyse_fba_progress.py generates progress graphs based on the .json summaries in /data.

It displays:
> Threshold progression over sessions (primary clinical metric)
> Accuracy (%) across sessions
> Run it with: python analyse_fba_progress.py

▶️ Running the training
Double-click the file:
cb_fba_training_psychopy.py
or run in terminal:
python cb_fba_training_psychopy.py
You will be prompted for:
angle set (0 = horizontal axis / UP–DOWN, 1 = vertical axis / LEFT–RIGHT)
stimulus eccentricity in degrees (H, V)
subject ID
Then the training session begins.

❗ Not Included (by design)

The original Matlab code contains an optional mechanism that automatically moves the stimulus deeper into the blind field after stable performance.
This has NOT been implemented here.
Stimulus location is chosen manually before each session.

📬 Acknowledgement
This work is a Python/PsychoPy reinterpretation of the structure of the original FBA training tool shared by the Huxlin Lab (Link: https://github.com/huxlinlab). 
It does not modify their scientific methods and aims to replicate their logic as closely as possible.

Feedback from the Rochester team is warmly welcomed.

Contact : 
Thomas Dietrich / thomas.a.dietrich@gmail.com

