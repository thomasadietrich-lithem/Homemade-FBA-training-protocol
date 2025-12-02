# FBA RDK Training — PsychoPy Implementation

A Python/PsychoPy reconstruction of the University of Rochester’s Feature-Based Attention (FBA) direction-range visual training task used in visual rehabilitation after occipital stroke.

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

- `cb_fba_training_psychopy.py`  
  Main training script implementing the FBA direction‐range RDK task.

- `cb_fba_training_psychopy_DR.py`  
  Alternative version explicitly configured for direction‐range training (10% signal dots, 90% noise).

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

---

## Requirements

- Python 3.10 – 3.12
- Dependencies:
  - PsychoPy 2024.2 or later
  - numpy
  - matplotlib (for analysis script)
  - json, csv, time, os (standard library)

PsychoPy can be downloaded from:

- https://www.psychopy.org/download.html

---

## Scientific fidelity – what is reproduced

The implementation aims to follow as closely as possible the behaviour of the Rochester Matlab + Psychtoolbox version:

- Random-dot kinematogram (RDK) with direction-range discrimination
- Three interleaved staircases with a 3-up / 1-down rule
- Same direction-range values as the Matlab code
- Same mapping between eccentricity (H, V in degrees) and pixels
- Same response rules (UP/DOWN vs LEFT/RIGHT depending on axis)
- Same structure for trial timing and feedback tones

This is not an official tool from the Huxlin Lab, but a reinterpretation based on their shared code and publications. Feedback from the original authors is very welcome.

---

## Direction-range RDK configuration

This PsychoPy implementation reproduces the direction-range (DR) version of the random-dot kinematogram task used in the Huxlin Lab FBA training studies.

Stimulus type:

- Circular random-dot kinematogram (RDK)
- Aperture radius: 2.5° visual angle
- Dot density: 3.5 dots/deg²
- Dot size: 14 arcmin (converted to pixels using monitor calibration)
- Dot speed: 10 deg/s
- Dot lifetime: 200 ms
- Stimulus duration: 500 ms
- Background: mid-gray

Direction-range logic and feature-based attention:

- On each trial, 10% of the dots move coherently in a direction that is slightly tilted upward or downward (for `angle_set = 0`) around the horizontal axis.
- The remaining 90% of the dots move in random directions within a direction range centred on the main motion axis.
- The direction range is expressed in degrees of spread around the axis (for example 85°, 53.1°, 33.2°, …, 0.5°).
- Large direction ranges (for example 85°) mean that dot directions are highly dispersed: the global motion signal is very weak, and the task is difficult.
- Small direction ranges (for example 2° or 0.5°) mean dot directions are tightly clustered: the global motion signal is strong, and the task is easier.
- The subject’s task is not to follow the average horizontal motion, but to report whether the global motion is tilted slightly upward or slightly downward.

Response mapping:

- angle_set = 0  
  Horizontal reference axis.  
  Task: discriminate upward vs downward tilt.  
  Response keys: Up and Down arrows.

- angle_set = 1  
  Vertical reference axis.  
  Task: discriminate leftward vs rightward tilt.  
  Response keys: Left and Right arrows.

---

## Staircase procedure

To closely follow the Rochester Matlab implementation, three interleaved staircases are used.

Direction-range levels (in degrees):


angle_range = [85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]
Initial staircase indices:

stair1 = 1 (hardest: 85°)
stair2 = 4 (intermediate: 20.75°)
stair3 = 8 (easier: 3.2°)


Behaviour:

On each trial, one of the three staircases is selected at random.

The current staircase index determines the direction range for that trial:
direction_range_deg = angle_range[stair_index - 1].

Each staircase follows a 3-up / 1-down rule:
after 3 consecutive correct responses in that staircase, its index is increased by 1 → task becomes more difficult (smaller direction range)
after 1 incorrect response in that staircase, its index is decreased by 1 → task becomes easier (larger direction range)
Indices are clipped to remain in the range 1 … len(angle_range).

Session threshold:

At the end of a session, the script converts the final staircase indices to degrees:

thr1 = angle_range[stair1_index - 1]
thr2 = angle_range[stair2_index - 1]
thr3 = angle_range[stair3_index - 1]

The overall session direction-range threshold is then defined as:
final_threshold_deg = (thr1 + thr2 + thr3) / 3
This is the same convention used in the original Matlab code.

Monitor calibration
Because the task is highly sensitive to spatial geometry, the script includes a basic calibration step.
On first run with a given monitor, the script asks for:
- physical screen width (cm)
- viewing distance (cm)
- screen resolution (pixels)

From these values it computes a conversion between degrees of visual angle (and arcminutes) and pixels, mimicking the logic in the Matlab/Psychtoolbox implementation.

The calibration is stored in:
monitor_profiles.json and is automatically reused on subsequent runs with the same monitor.
This is important if training is performed on different computers or screens.

Analytics and dashboard
The file analyse_fba_progress.py reads all *_summary.json files in the data/ directory and builds an analysis dashboard.

For each condition (defined by the combination of task, H_deg, V_internal and angle_set), it shows:

Left panel:
- the three staircase thresholds across sessions (in degrees)
- the mean threshold across staircases (thicker line)
- a 3-session moving average of the mean threshold
- a linear regression line for the mean threshold, with slope printed in deg/session

Right panel:
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
analysis_outputs/fba_dashboard.png

This is intended to approximate the kind of plots shown in the Huxlin Lab publications, and to help decide when a given training location has reached a plateau (for example when the mean threshold trend is flat over many sessions) and might be ready to shift further into the blind field.

Running the training
To run a training session:

Double-click cb_fba_training_psychopy.py in PsychoPy
or run:

python cb_fba_training_psychopy.py
You will be prompted for:

angle_set (0 = horizontal axis / Up–Down, 1 = vertical axis / Left–Right)
stimulus eccentricity in degrees (H and V)
subject ID

To analyse your progress:

python analyse_fba_progress.py
This will read all existing summary files and update the dashboard.

Acknowledgement and disclaimer
This project is a Python/PsychoPy reinterpretation of the structure and logic of the original FBA training tools shared by the Huxlin Lab (University of Rochester):
https://github.com/huxlinlab
It is not an official clinical tool and should not be used as a substitute for medical advice or supervised rehabilitation. Any use for self-training should be discussed with a qualified clinician.

Feedback or corrections from the original authors, or from researchers familiar with the protocol, are very welcome.

Contact:
Thomas Dietrich – thomas.a.dietrich@gmail.com
