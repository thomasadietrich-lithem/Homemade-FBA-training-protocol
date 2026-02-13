# FBA RDK Training — PsychoPy Implementation

A Python/PsychoPy reconstruction of the University of Rochester’s Feature-Based Attention (FBA) random-dot visual training task used in visual rehabilitation after occipital stroke.

This repository provides **two training scripts (two RDK variants)**:

1. **Global Tilt** (`cb_fba_training_psychopy_TILT_GLOBAL.py`)  
   **No dispersion**: on each trial, **all dots move in exactly the same direction**. The participant reports the **global tilt direction** (up vs down, or left vs right depending on axis). Difficulty is controlled by the staircase via **tilt magnitude**.

2. **Direction Range (DR)** (`cb_fba_training_psychopy_DR.py`)  
   **With dispersion**: on each trial, **each dot’s direction is sampled around a main motion axis** by adding **Gaussian angular noise**. Difficulty is controlled by the staircase via the **amount of angular dispersion** (larger dispersion = harder).

This implementation was reverse-engineered from the Matlab + Psychtoolbox code shared by the Huxlin Lab (University of Rochester), with the aim of preserving:

- stimulus geometry
- staircase dynamics
- dot motion parameters
- timing
- eccentricity mapping
- response rules
- auditory feedback

It is intended for personal training and experimentation. It does not depend on Matlab, Psychtoolbox, AWS, or any cloud infrastructure.

---

## Files in this repository

- `cb_fba_training_psychopy_TILT_GLOBAL.py`  
  Training script implementing the **Global Tilt** variant (**100% coherent motion**, report tilt direction).

- `cb_fba_training_psychopy_DR.py`  
  Training script implementing the **Direction Range (DR)** variant (**per-dot angular dispersion** around the main axis via Gaussian noise).

- `analyse_fba_progress.py`  
  Analysis script that reads session summary JSON files and produces a dashboard showing threshold and accuracy over sessions, including the three staircases.

Two additional things are created automatically when you run the program:

1. `data/` directory  
   Contains:
   - trial-level `.csv` logs
   - session-level `*_summary.json` files (one per training session)

2. `monitor_profiles.json`  
   Stores monitor calibration so that stimulus geometry remains consistent across different screens.

You do not need to create these manually.

-----

## Requirements

- Python 3.10 – 3.12
- Dependencies:
  - PsychoPy 2024.2 or later
  - numpy
  - matplotlib (for analysis script)
  - json, csv, time, os (standard library)

PsychoPy can be downloaded from:

- https://www.psychopy.org/download.html

_____

## Scientific fidelity – what is reproduced

The implementation aims to follow as closely as possible the behaviour of the Rochester Matlab + Psychtoolbox version, while providing two stimulus modes:

- **Global Tilt** (fully coherent motion; discrimination of tilt direction)
- **Direction Range (DR)** (angular dispersion around a main axis; discrimination under direction noise)
- Three interleaved staircases with a 3-up / 1-down rule
- Same eccentricity mapping between degrees (H, V) and pixels
- Same response mapping (UP/DOWN vs LEFT/RIGHT depending on axis)
- Same structure for trial timing and feedback tones

This is not an official tool from the Huxlin Lab, but a reinterpretation based on their shared code and publications. Feedback from the original authors is very welcome.

---

## Global Tilt variant (`cb_fba_training_psychopy_TILT_GLOBAL.py`)

### Stimulus type

- Circular random-dot kinematogram (RDK)
- Aperture radius: 2.5° visual angle
- Dot density: 3.5 dots/deg²
- Dot size: 14 arcmin (converted to pixels using monitor calibration)
- Dot speed: 10 deg/s
- Dot lifetime: 200 ms
- Stimulus duration: 500 ms
- Background: mid-gray

### Global-tilt logic

- On each trial, **100% of the dots move in the same direction** (no direction dispersion).
- Motion is tilted slightly to one of two alternatives:
  - around the **horizontal reference axis**: upward vs downward tilt
  - around the **vertical reference axis**: leftward vs rightward tilt
- Task difficulty is controlled by the staircase via **tilt magnitude** (smaller tilt = harder).

### Response mapping

- angle_set = 0  
  Horizontal reference axis.  
  Task: discriminate upward vs downward tilt.  
  Response keys: Up and Down arrows.

- angle_set = 1  
  Vertical reference axis.  
  Task: discriminate leftward vs rightward tilt.  
  Response keys: Left and Right arrows.

-----

## Direction Range (DR) variant (`cb_fba_training_psychopy_DR.py`)

### Stimulus type

- Circular random-dot kinematogram (RDK)
- Aperture radius: 2.5° visual angle
- Dot density: 3.5 dots/deg²
- Dot size: 14 arcmin (converted to pixels using monitor calibration)
- Dot speed: 10 deg/s
- Dot lifetime: 200 ms
- Stimulus duration: 500 ms
- Background: mid-gray

### Direction-range / dispersion logic

- On each trial, dots move around a **main motion axis** (horizontal or vertical).
- **Each dot’s direction is jittered** by adding **Gaussian angular noise** around the main axis (dot-by-dot dispersion).
- The dispersion magnitude is controlled by the staircase via values in `angle_range`:
  - **larger dispersion** → global motion signal is weaker → harder
  - **smaller dispersion** → directions are more clustered → easier

### Response mapping

- angle_set = 0  
  Horizontal reference axis.  
  Task: discriminate upward vs downward tilt.  
  Response keys: Up and Down arrows.

- angle_set = 1  
  Vertical reference axis.  
  Task: discriminate leftward vs rightward tilt.  
  Response keys: Left and Right arrows.

_____

## Staircase procedure

To closely follow the Rochester Matlab implementation, three interleaved staircases are used.

### Levels (in degrees)

angle_range = [85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]

### Initial staircase indices

stair1 = 1 (hardest: 85°)  
stair2 = 4 (intermediate: 20.75°)  
stair3 = 8 (easier: 3.2°)

### Behaviour

On each trial, one of the three staircases is selected at random.

- In **DR mode**, the staircase index determines the **dispersion** for that trial:  
  direction_range_deg = angle_range[stair_index - 1]

- In **Global Tilt mode**, the staircase index determines the **tilt magnitude** for that trial:  
  tilt_deg = angle_range[stair_index - 1]

Each staircase follows a 3-up / 1-down rule:  
after 3 consecutive correct responses in that staircase, its index is increased by 1 → task becomes more difficult (smaller range / smaller tilt)  
after 1 incorrect response in that staircase, its index is decreased by 1 → task becomes easier (larger range / larger tilt)  
Indices are clipped to remain in the range 1 … len(angle_range).

### Session threshold

At the end of a session, the script converts the final staircase indices to degrees:

thr1 = angle_range[stair1_index - 1]  
thr2 = angle_range[stair2_index - 1]  
thr3 = angle_range[stair3_index - 1]

The overall session threshold is then defined as:

final_threshold_deg = (thr1 + thr2 + thr3) / 3

This is the same convention used in the original Matlab code.

-----

## Monitor calibration

Because the task is highly sensitive to spatial geometry, the script includes a basic calibration step.  
On first run with a given monitor, the script asks for:

- physical screen width (cm)
- viewing distance (cm)
- screen resolution (pixels)

From these values it computes a conversion between degrees of visual angle (and arcminutes) and pixels, mimicking the logic in the Matlab/Psychtoolbox implementation.

The calibration is stored in:  
`monitor_profiles.json` and is automatically reused on subsequent runs with the same monitor.

This is important if training is performed on different computers or screens.

---

## Analytics and dashboard

The file `analyse_fba_progress.py` reads all `*_summary.json` files in the `data/` directory and builds an analysis dashboard.

For each condition (defined by the combination of task, H_deg, V_internal and angle_set), it shows:

### Left panel

- the three staircase thresholds across sessions (in degrees)
- the mean threshold across staircases (thicker line)
- a 3-session moving average of the mean threshold
- a linear regression line for the mean threshold, with slope printed in deg/session

### Right panel

- accuracy (%) across sessions
- a 3-session moving average of accuracy
- a linear regression line for accuracy, with slope printed in %/session

The script:

- groups sessions by condition (so different visual field locations appear as different rows)
- orders sessions chronologically by timestamp
- prints a text summary to the console, including:
  - mean threshold start / last
  - staircase thresholds for each session
  - accuracy start / last
  - linear slopes and R² for threshold and accuracy
  - a simple “recent trend” over the last five sessions, to detect plateaus

The dashboard figure is also saved as:  
`analysis_outputs/fba_dashboard.png`

This is intended to approximate the kind of plots shown in the Huxlin Lab publications, and to help decide when a given training location has reached a plateau and might be ready to shift further into the blind field.

_____

## Running the training

To run a training session:

Double-click `cb_fba_training_psychopy_TILT_GLOBAL.py` or `cb_fba_training_psychopy_DR.py` in PsychoPy  
or run:

python cb_fba_training_psychopy_TILT_GLOBAL.py
python cb_fba_training_psychopy_DR.py

_____

You will be prompted for:

- subject ID
- angle_set (0 = horizontal axis / Up–Down, 1 = vertical axis / Left–Right)
- stimulus eccentricity in degrees (H and V)
- number of trials

To analyse your progress:
python analyse_fba_progress.py

This will read all existing summary files and update the dashboard.

_____
### Acknowledgement and disclaimer

This project is a Python/PsychoPy reinterpretation of the structure and logic of the original FBA training tools shared by the Huxlin Lab (University of Rochester):
https://github.com/huxlinlab It is not an official clinical tool and should not be used as a substitute for medical advice or supervised rehabilitation. Any use for self-training should be discussed with a qualified clinician.

Feedback or corrections from the original authors, or from researchers familiar with the protocol, are very welcome.

Contact:
Thomas Dietrich – thomas.a.dietrich@gmail.com
