# FBA Training — Tilt Global (PsychoPy Implementation)

A Python/PsychoPy implementation of a Feature-Based Attention (FBA) training task inspired by the Huxlin Lab (University of Rochester), designed for home-based fine direction discrimination (FDD) training.

This repository provides:

- a **training script** (Tilt Global paradigm)
- an **analysis dashboard** (Weibull-based, longitudinal, Huxlin-style)

The objective is to preserve the **scientific structure of the original paradigm**, while providing a fully standalone workflow (no Matlab, no Psychtoolbox).

---

# Repository structure

This repository contains **two main scripts**:

## 1. `FBA_Training_Tilt_Global.py`

Standalone PsychoPy training script implementing:

- a **coherent-motion random dot stimulus (RDK)**
- a **Tilt Global paradigm (100% coherence)**
- a **triangular central informative pre-cue**
- **3 interleaved staircases (fixed, non-configurable)**

---

## 2. `FBA_Training_Tilt_Global_Analytics.py`

Standalone analysis script that:

- reads all session CSV files
- performs **Weibull psychometric fitting**
- builds a **multi-layer interactive dashboard**
- exports:
  - plots (PNG)
  - structured summaries (JSON, CSV)
  - a **navigable HTML dashboard**

---

# Training paradigm — Tilt Global

## Stimulus

- Circular random-dot kinematogram (RDK)
- Aperture radius: **2.5°**
- Dot density: **3.5 dots/deg²**
- Dot size: **14 arcmin**
- Dot speed: **10 deg/s**
- Dot lifetime: **250 ms**
- Stimulus duration: **500 ms**

---

## Core principle

Each trial:

- **100% of dots move in the same direction**
- Motion is slightly tilted around a reference axis

Two task modes:

| angle_set | Reference axis | Task |
|----------|----------------|------|
| 0        | Horizontal     | Up vs Down tilt |
| 1        | Vertical       | Left vs Right tilt |

---

## Difficulty control

Difficulty is controlled via **tilt magnitude**:


[85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]


- large angle → easy  
- small angle → difficult  

---

## Staircase procedure

- **3 interleaved staircases (fixed)**
- **3-up / 1-down rule**

Behavior:

- 3 correct → harder (↓ angle)
- 1 incorrect → easier (↑ angle)

---

## Pre-cue

- Triangular central cue
- Encodes:
  - **motion direction**
  - **task difficulty (angular span)**

---

## Output data

Each session produces:

### `data/` folder

- `*_trials.csv` → trial-level data  
- `*_summary.json` → session summary  

---

# Monitor calibration

On first run:

User inputs:

- screen width (cm)
- viewing distance (cm)

Stored in:


monitor_profiles.json


Used to convert:

- degrees → pixels  
- arcmin → pixels  

---

# Analysis pipeline

## Script: `FBA_Training_Tilt_Global_Analytics.py`

This script performs a **Huxlin-style analysis**.

---

## 1. Psychometric fitting

- Weibull fit in:


log10(angle + 1)


Constants:

- chance = 0.5  
- lapse = 0.05  
- threshold target = 0.725  

---

## 2. Outputs

### Per session

- psychometric curve (Weibull)  
- threshold estimate (deg)  

Fit quality metrics:

- log-likelihood  
- pseudo-R²  
- AIC / BIC  

---

### Longitudinal

- threshold across sessions  
- accuracy across sessions  
- **3-session pooled threshold**  
- variability (standard deviation)

---

### Within-session

- performance split into blocks  

---

### Stimulus-level

- performance per direction (UP/DOWN or LEFT/RIGHT)

---

## 3. Dashboard

Generated automatically:


analysis_results/dashboard.html


Features:

- interactive navigation  
- session comparison  
- Weibull overlay  
- smoothing controls:
  - raw  
  - moving average (5 / 10)  
  - pooled threshold  

---

## 4. PDF export

From browser:


Export / print as PDF


Includes:

- optimized layout (A4)  
- non-breaking figures  
- clean pagination  

---

# Running the training

```bash
python FBA_Training_Tilt_Global.py

User inputs:

subject ID
angle_set
stimulus location (H, V)
number of trials
Running the analysis
python FBA_Training_Tilt_Global_Analytics.py

The script:

scans /data
processes all sessions
generates dashboard
opens it automatically in browser
Scientific references

The implementation is inspired by the following key publications from the Huxlin Lab and related work:

1. Huxlin et al., 2009

Perceptual relearning of complex visual motion after V1 damage in humans
https://www.jneurosci.org/content/29/13/3981

→ Demonstrates recovery of motion perception through training after occipital stroke.

2. Das et al., 2014

Rehabilitation of visual function in cortical blindness
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4097947/

→ Reviews mechanisms and protocols for visual rehabilitation, including motion training.

3. Cavanaugh et al., 2019

Visual training improves perceptual performance in cortically blind fields
https://www.frontiersin.org/articles/10.3389/fnsys.2019.00036/full

→ Provides detailed training paradigms and evidence for long-term improvements.

Scientific fidelity

This implementation follows:

TrainingCodes lineage (stimulus + staircase)
DataFitting lineage (Weibull analysis)

Key preserved elements:

fixed 3 staircases
log-scale difficulty
stimulus geometry
timing
response mapping
Scope and limitations

This is:

a home training tool
not a clinical or eye-tracked system

It does NOT include:

eye tracking
fixation control
clinical validation
Acknowledgement

Inspired by:

Huxlin Lab — University of Rochester
https://github.com/huxlinlab

Disclaimer

This is not a medical device.

Use in rehabilitation should be supervised by a qualified clinician.

Contact

Thomas Dietrich
thomas.a.dietrich@gmail.com
