# FBA Training Tilt Global

Python/PsychoPy implementation of a Huxlin-lab-inspired **Feature-Based Attention (FBA)** home-training task for **fine direction discrimination (FDD)** using a coherent-motion random-dot stimulus, plus a dedicated analytics dashboard based on **Weibull psychometric fitting**.


- `FBA_Training_Tilt_Global.py`
- `FBA_Training_Tilt_Global_Analytics.py`


---

## Table of contents

- [Repository structure](#repository-structure)
- [Application 1 — FBA_Training_Tilt_Global.py](#application-1--fba_training_tilt_globalpy)
- [Application 2 — FBA_Training_Tilt_Global_Analytics.py](#application-2--fba_training_tilt_global_analyticspy)
- [Paradigm summary](#paradigm-summary)
- [Pre-cue mechanism](#pre-cue-mechanism)
- [Adaptive staircase](#adaptive-staircase)
- [Weibull psychometric analysis](#weibull-psychometric-analysis)
- [Longitudinal dashboard](#longitudinal-dashboard)
- [Installation](#installation)
- [Tutorial](#tutorial)
- [Research references](#research-references)
- [Acknowledgement and disclaimer](#acknowledgement-and-disclaimer)

---

## Repository structure

The repository is expected to contain the following two Python files:

```text
.
├── FBA_Training_Tilt_Global.py
└── FBA_Training_Tilt_Global_Analytics.py
```

After running sessions and analyses, the following folders/files are created automatically:

```text
.
├── FBA_Training_Tilt_Global.py
├── FBA_Training_Tilt_Global_Analytics.py
├── monitor_profiles.json
├── data/
│   ├── <subject>_FBA_TILTGLOBAL_<timestamp>_trials.csv
│   └── <subject>_FBA_TILTGLOBAL_<timestamp>_summary.json
└── analysis_results/
    ├── dashboard.html
    ├── session_summary_table.csv
    ├── analysis_dashboard_summary.json
    ├── *_psychometric_weibull.png
    ├── *_session_summary.json
    ├── *_staircase_trace.png
    ├── *_stimulus_distribution.png
    ├── longitudinal_*_accuracy_threshold_overlay.png
    ├── recent_*_within_session_block_accuracy.png
    └── longitudinal_*_stimulus_class_accuracy.png
```

Important distinction:

- The **training application** creates and fills the `data/` folder.
- The **analytics application** reads `data/*_trials.csv` and writes results to `analysis_results/`.
- If trial CSV files are not inside a folder named `data`, the analytics application can also search next to the script and write `analysis_results/` next to the input files. The analytics application will automaticaly open an HTML dashboard version in your default web browser.  

---

## Application 1 — `FBA_Training_Tilt_Global.py`

### Purpose

`FBA_Training_Tilt_Global.py` runs the actual PsychoPy home-training session.

It implements a coherent-motion Random Dot Kinematogram (RDK) task with a triangular feature-based attention pre-cue.

The task is designed for **fine direction discrimination**:

- all dots move coherently in the same direction on a given trial;
- the direction is slightly tilted relative to a horizontal or vertical reference axis;
- the participant reports the direction of the tilt using arrow keys;
- difficulty is controlled by the angular deviation from the reference direction.

This version is a **home-training implementation**. It does not perform eye tracking.

---

### Training lineage

The training script explicitly follows a single-source lineage:

- training/stimulus branch: `huxlinlab/TrainingCodes`
- downstream fitting branch: `huxlinlab/DataFitting`

The implementation intentionally uses:

- fixed 3 interleaved staircases;
- the TrainingCodes fine-difficulty scale;
- dot density of 3.5 dots/deg²;
- a triangular central informative pre-cue;
- extensive inline comments for auditability.

---

### User inputs

At launch, the script opens a PsychoPy dialog asking for:

| Input | Meaning |
|---|---|
| `Subject ID` | Subject identifier used in output filenames |
| `Angle set` | `0 = Horizontal UP/DOWN`, `1 = Vertical LEFT/RIGHT` |
| `Stimulus horizontal eccentricity` | Horizontal visual-field position in degrees, Left(-) to Right(+) |
| `Stimulus vertical eccentricity` | Vertical visual-field position in degrees, Down(-) to Up(+) |
| `Number of Trials` | Number of trials in the session |
| `Show pre-cue?` | Enables or disables the triangular informative pre-cue |

---


### Monitor calibration

The script asks the user to provide:

- screen width in centimeters;
- viewing distance in centimeters.

Default values:

```text
screen width = 61.0 cm
viewing distance = 42.0 cm
```

The script then computes:

```text
arcmin_per_pixel
```

This value is used to convert visual degrees and arcminutes into pixels.

The calibration is saved in:

```text
monitor_profiles.json
```

The profile key depends on:

```text
operating system + screen resolution
```

If screen detection fails, the script falls back to:

```text
1920 × 1080 px
```

---

### Stimulus parameters

The current implementation uses the following fixed stimulus parameters:

| Parameter | Value |
|---|---:|
| Aperture radius | 2.5° |
| Aperture diameter | 5.0° |
| Dot density | 3.5 dots/deg² |
| Approximate dot count | 69 dots |
| Dot speed | 10 deg/s |
| Dot size | 14 arcmin |
| Dot lifetime | 250 ms |
| Stimulus duration | 500 ms |
| Background RGB | 0.5 |
| Dot color RGB | -1.0 |
| Fixation outer radius | 0.10° |
| Fixation inner radius | 0.05° |

The approximate dot count is computed from the aperture area:

```text
round(3.5 × pi × 2.5²) = approximately 69 dots
```

---

### Trial timing

Each trial follows this timing sequence:

| Phase | Duration |
|---|---:|
| Fixation | 1.0 s |
| Pre-cue | 0.2 s |
| Post-cue ISI | 0.5 s |
| RDK stimulus | 500 ms |
| Feedback / inter-trial interval | approximately 0.5 s |

The actual number of stimulus frames is computed from the detected monitor refresh rate. If PsychoPy cannot estimate the refresh rate, the script falls back to 60 Hz.

---

### Response mapping

The task has two possible angle sets.

#### `angle_set = 0` — Horizontal reference axis

The stimulus is based on horizontal motion directions.

The participant reports whether the motion is tilted:

- upward;
- downward.

Response keys:

```text
UP arrow   = motion tilted upward
DOWN arrow = motion tilted downward
```

#### `angle_set = 1` — Vertical reference axis

The stimulus is based on vertical motion directions.

The participant reports whether the motion is tilted:

- leftward;
- rightward.

Response keys:

```text
LEFT arrow  = motion tilted leftward
RIGHT arrow = motion tilted rightward
```

---

### Motion generation

This version is **Tilt Global only**.

On each trial:

- all dots share the same motion vector;
- no angular dispersion is added;
- no Direction Range (DR) condition exists in this repository;
- difficulty is controlled only by `angle_dev_deg`.

The script stores the actual stimulus angle in:

```text
angle_deg
```

The script stores the difficulty level in:

```text
angle_dev_deg
```

---

## Application 2 — `FBA_Training_Tilt_Global_Analytics.py`

### Purpose

`FBA_Training_Tilt_Global_Analytics.py` is the analysis and dashboard application.

It performs:

- trial-level CSV discovery;
- data cleaning;
- one Weibull psychometric fit per session;
- one longitudinal dashboard per trained location;
- recent within-session block accuracy plots;
- stimulus-class comparisons;
- interactive HTML dashboard generation.

It is explicitly descriptive:

```text
No automatic clinical decision rule is implemented.
```

---



### Output folder

When CSV files live in a `data/` folder, the analytics tool writes outputs to:

```text
analysis_results/
```

next to the `data/` folder.

Example:

```text
project/
├── data/
│   └── *_trials.csv
└── analysis_results/
    └── dashboard.html
```

If the input CSV files are not inside `data/`, the tool creates `analysis_results/` next to the input CSV files.

---

### Analytics outputs

For each session, the analytics application can generate:

| Output | Description |
|---|---|
| `*_psychometric_weibull.png` | Session psychometric curve with Weibull fit |
| `*_session_summary.json` | Session-level analysis summary |
| `*_staircase_trace.png` | Trial-by-trial staircase trace |
| `*_stimulus_distribution.png` | Distribution of trials across stimulus levels |

Across sessions and locations, it generates:

| Output | Description |
|---|---|
| `session_summary_table.csv` | Table of all analyzed sessions |
| `analysis_dashboard_summary.json` | Structured dashboard summary |
| `dashboard.html` | Main navigable HTML dashboard |
| `longitudinal_*_accuracy_threshold_overlay.png` | Accuracy and threshold over sessions |
| `recent_*_within_session_block_accuracy.png` | Within-session block accuracy for recent sessions |
| `longitudinal_*_stimulus_class_accuracy.png` | UP/DOWN or LEFT/RIGHT accuracy comparison |

The HTML dashboard includes:

- summary view;
- per-location sections;
- session tables;
- psychometric Weibull plots;
- staircase traces;
- stimulus-level distributions;
- interactive session comparison;
- browser-based export / print to PDF via `window.print()`.

---

## Paradigm summary

This repository implements a coherent-motion fine direction discrimination paradigm.

The participant fixates centrally while a peripheral coherent-motion RDK is presented at a user-specified visual-field location.

Depending on `angle_set`, the participant discriminates either:

- upward vs downward tilt around a horizontal motion axis;
- leftward vs rightward tilt around a vertical motion axis.

The task is adaptive: correct and incorrect responses update one of three interleaved staircases.

---

## Pre-cue mechanism

The pre-cue is a triangular central cue presented at fixation before the RDK stimulus.

It is intended to implement a feature-based attention signal.

### Geometry

The cue:

- is drawn at fixation;
- has a fixed depth of approximately 1°;
- has a width that depends on trial difficulty;
- has a minimum span to avoid collapsing visually.

Cue-width rule:

```text
half_span_deg = max(angle_deviation_deg, 5.0) / 60
```

The constant `5.0` is the current minimum cue span in degrees of the difficulty scale, converted into a small visual-angle width by dividing by 60.

### Cue orientation

For `angle_set = 0`:

- the cue points toward the rightward or leftward base direction.

For `angle_set = 1`:

- the cue points toward the downward or upward base direction.

The participant still responds to the fine tilt direction:

- UP/DOWN for `angle_set = 0`;
- LEFT/RIGHT for `angle_set = 1`.

### Practical role

The pre-cue is intended to:

- orient feature-based attention before stimulus onset;
- reduce uncertainty about the relevant motion feature;
- help the participant prepare for the upcoming RDK stimulus.

---

## Adaptive staircase

The current version uses:

```text
3 fixed interleaved staircases
```

The staircase identities are:

```text
1, 2, 3
```

The initial zero-based staircase indices are:

```text
[0, 3, 7]
```

Corresponding initial difficulty levels:

```text
85.0°, 20.75°, 3.2°
```

### Difficulty scale

The difficulty scale is:

```text
[85.0, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]
```

Because the list is descending:

- lower index = larger angular deviation = easier trial;
- higher index = smaller angular deviation = harder trial.

### Update rule

The script implements a 3-correct-down / 1-incorrect-up adaptive rule.

For each staircase independently:

- after 3 consecutive correct responses:
  - staircase index increases by 1;
  - angular deviation becomes smaller;
  - the task becomes harder;

- after 1 incorrect response:
  - staircase index decreases by 1;
  - angular deviation becomes larger;
  - the task becomes easier.

Indices are clipped to stay inside the difficulty scale.

### Coarse session estimate

At the end of the session, the training script computes:

```text
coarse_session_staircase_estimate_deg =
mean(current level of staircase 1,
     current level of staircase 2,
     current level of staircase 3)
```

This value is saved in the session JSON. It is a coarse staircase estimate, not the final Weibull threshold.

---

## Weibull psychometric analysis

The analytics application estimates a psychometric threshold from trial-level data using a Weibull function.

### Fitted stimulus space

The fit is performed in log-transformed stimulus space:

```text
x = log10(angle_dev_deg + 1)
```

This follows the DataFitting-compatible workflow used by the analysis script.

### Weibull function

The psychometric function used by the analytics application is:

```text
p(correct) = chance + (1 - chance - lapse) × (1 - exp(-(x / alpha)^beta))
```

Where:

| Symbol | Meaning |
|---|---|
| `x` | `log10(angle_dev_deg + 1)` |
| `alpha` | threshold parameter in log space |
| `beta` | slope parameter |
| `chance` | chance-level performance |
| `lapse` | lapse-rate parameter |

Current constants:

```text
chance = 0.5
lapse = 0.05
threshold target = 0.725
```

The threshold target of `0.725` corresponds to 72.5% correct.

### Threshold conversion

After fitting in log space, the threshold is converted back into degrees.

The analytics application computes:

```text
scaled = (threshold_target - chance) / (1 - chance - lapse)

x_threshold_log =
alpha × (-log(1 - scaled))^(1 / beta)

threshold_deg =
10^(x_threshold_log) - 1
```

The resulting `threshold_deg` is the refined session threshold.

### Fit validation

The analytics script only fits sessions that meet minimum data requirements:

```text
minimum trials = 80
minimum unique stimulus levels = 4
```

If these criteria are not met, the fit is marked invalid.

The script also rejects pathological thresholds:

```text
threshold_deg must be finite
threshold_deg must not exceed max(observed_max × 1.5, observed_max + 5)
```

This avoids fabricating extreme or numerically divergent thresholds.

### Fit-quality metrics

The analytics dashboard computes descriptive fit-quality metrics:

- fitted log-likelihood;
- null log-likelihood;
- pseudo-R²;
- AIC;
- BIC.

These metrics are descriptive only.

They do not change the threshold and do not trigger any automatic clinical decision.

---

## Longitudinal dashboard

The dashboard groups sessions by trained location.

A location ID is inferred from the companion JSON summary:

```text
H<value>_V<value>
```

Example:

```text
H+3.00_V-2.00
```

For each trained location, the dashboard shows:

- session-by-session Weibull threshold;
- raw accuracy;
- 3-session mean threshold;
- 3-session mean accuracy;
- 3-session pooled Weibull threshold;
- within-session block accuracy for recent sessions;
- stimulus-class accuracy.

### Within-session block analysis

Each session is split into:

```text
6 equal trial blocks
```

The dashboard plots accuracy across these blocks for the most recent sessions.

Current recent-session window:

```text
5 sessions
```

### Stimulus-class analysis

The analytics application classifies trials by stimulus direction:

For `angle_set = 0`:

```text
UP vs DOWN
```

For `angle_set = 1`:

```text
LEFT vs RIGHT
```

This helps detect asymmetric performance across response categories.

---



## Installation

Recommended Python version:

```text
Python 3.10 – 3.12
```

Required Python packages:

```text
psychopy
numpy
pandas
matplotlib
scipy
```

The training application requires PsychoPy.

The analytics application requires:

```text
numpy
pandas
matplotlib
scipy
```

---

## Tutorial

### 1. Clone or download the repository

The repository should contain only:

```text
FBA_Training_Tilt_Global.py
FBA_Training_Tilt_Global_Analytics.py
```

### 2. Install dependencies

Using pip:

```bash
pip install numpy pandas matplotlib scipy psychopy
```

PsychoPy can also be installed from the official PsychoPy distribution if preferred.

### 3. Run the training application

From a terminal:

```bash
python FBA_Training_Tilt_Global.py
```

Or open the file in PsychoPy and run it from the PsychoPy interface.

### 4. Complete the setup dialog

You will be asked for:

```text
Subject ID
Angle set
Horizontal eccentricity
Vertical eccentricity
Number of trials
Show pre-cue?
```

If no monitor profile exists, you will also be asked for:

```text
screen width in cm
viewing distance in cm
```

### 5. Complete the task

During the task:

- keep fixation on the central dot;
- respond with the relevant arrow keys;
- press `ESC` to abort if needed.

### 6. Locate the saved data

After the session, the application saves data to:

```text
data/
```

You should see:

```text
*_trials.csv
*_summary.json
```

### 7. Run the analytics application

From the same project folder:

```bash
python FBA_Training_Tilt_Global_Analytics.py
```

The script will search for:

```text
data/*_trials.csv
```

Then it will generate:

```text
analysis_results/
```


## Research references

This repository is inspired by the Huxlin Lab’s visual rehabilitation work and public code lineage.

Recommended references:

1. Huxlin, K. R., Martin, T., Kelly, K., Riley, M., Friedman, D. I., Burgin, W. S., & Hayhoe, M. (2009).  
   **Perceptual Relearning of Complex Visual Motion after V1 Damage in Humans.**  
   *Journal of Neuroscience, 29(13), 3981–3991.*  
   https://www.jneurosci.org/content/29/13/3981

2. Das, A., Tadin, D., & Huxlin, K. R. (2014).  
   **Beyond Blindsight: Properties of Visual Relearning in Cortically Blind Fields.**  
   *Journal of Neuroscience, 34(35), 11652–11664.*  
   https://www.jneurosci.org/content/34/35/11652

3. Cavanaugh, M. R., Barbot, A., Carrasco, M., & Huxlin, K. R. (2019).  
   **Feature-based attention potentiates recovery of fine direction discrimination in cortically blind patients.**  
   *Neuropsychologia, 128, 315–324.*  
   https://doi.org/10.1016/j.neuropsychologia.2017.12.010](https://pmc.ncbi.nlm.nih.gov/articles/PMC5994362/)

Public Huxlin Lab GitHub organization:

```text
https://github.com/huxlinlab
```


## Acknowledgement and disclaimer

This project is a Python/PsychoPy reinterpretation of the structure and logic of the original FBA training tools shared by the Huxlin Lab (University of Rochester): https://github.com/huxlinlab

It is not an official clinical tool and should not be used as a substitute for medical advice or supervised rehabilitation. Any use for self-training should be discussed with a qualified clinician.

Feedback or corrections from the original authors, or from researchers familiar with the protocol, are very welcome.

Contact: Thomas Dietrich – thomas.a.dietrich@gmail.com
