"""
Standalone PsychoPy implementation of a Huxlin-lab-inspired home training task for
fine direction discrimination (FDD) of a coherent-motion random-dot stimulus with a
feature-based attention (FBA) pre-cue.

Design choices in this finalized version
---------------------------------------
This file intentionally follows a single source lineage for the training side:
- stimulus/training branch: huxlinlab/TrainingCodes
- downstream fitting branch: huxlinlab/DataFitting

Accordingly, this script uses:
- fixed 3 interleaved staircases (not user-configurable)
- the current TrainingCodes fine-difficulty scale
- dot density = 3.5 dots/deg²
- a triangular central pre-cue matching the previous working implementation
- extensive inline comments for auditability

This is a HOME TRAINING script. It is not an eye-tracked laboratory verification tool.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import platform
import tempfile
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from psychopy import core, event, gui, monitors, sound, visual


# -----------------------------------------------------------------------------
# Constants chosen to stay within the TrainingCodes lineage
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRNAME = "data"
MONITOR_CACHE_FILE = os.path.join(BASE_DIR, "monitor_profiles.json")

# The Huxlin home-training branch uses a quasi-logarithmic scale with 12 levels.
# In the MATLAB code these are loaded from HuxlinLabFBASetup.m and then used by
# HuxlinLabFBAtraining.m. We hard-code them here so the protocol is explicit.
ANGLE_RANGE_DEG: List[float] = [85.0, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]

# The protocol is intentionally NOT exposed as generic/flexible at the UI level.
# Three interleaved staircases are part of the design.
N_STAIRCASES = 3
INITIAL_STAIR_IDXS = [0, 3, 7]  # MATLAB {1,4,8} in 1-based indexing.
TOTAL_TRIALS_DEFAULT = 300

# Stimulus geometry/timing. These values are kept stable and explicit so that
# longitudinal comparisons remain interpretable.
VIEWING_DISTANCE_CM_DEFAULT = 42.0
SCREEN_WIDTH_CM_DEFAULT = 61.0
FIXATION_DURATION_S = 1.0
CUE_DURATION_S = 0.2
POST_CUE_ISI_S = 0.5
STIMULUS_DURATION_MS = 500.0
DOT_LIFETIME_MS = 250.0
APERTURE_RADIUS_DEG = 2.5  # 5° diameter aperture
DOT_DENSITY_PER_DEG2 = 3.5
DOT_SPEED_DEG_PER_S = 10.0
DOT_SIZE_ARCMIN = 14.0
BACKGROUND_RGB = 0.5
DOT_COLOR_RGB = -1.0
FIX_OUTER_RADIUS_DEG = 0.10
FIX_INNER_RADIUS_DEG = 0.05

# Difficulty floor for cue width in the MATLAB branch: the cue should not collapse
# to an invisibly small extent at the finest levels.
MIN_CUE_SPAN_DEG = 5.0


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def get_screen_pixels() -> Tuple[int, int]:
    """Best-effort screen pixel query.

    PsychoPy already knows the window size once a full-screen window is opened, but
    we need a reasonable default before that. If pyglet fails, we fall back to a
    conventional 1920×1080 assumption and still let the user override physical size.
    """
    try:
        import pyglet  # Imported lazily so syntax-checking does not depend on it.

        display = pyglet.canvas.get_display()
        screen = display.get_default_screen()
        return int(screen.width), int(screen.height)
    except Exception:
        return 1920, 1080


@dataclass
class MonitorGeometry:
    width_cm: float
    distance_cm: float
    res_x: int
    res_y: int
    arcmin_per_pix: float



def deg_to_pix(deg: float, geom: MonitorGeometry) -> float:
    return (deg * 60.0) / geom.arcmin_per_pix



def pix_to_deg(px: float, geom: MonitorGeometry) -> float:
    return (px * geom.arcmin_per_pix) / 60.0



def _safe_json_load(path: str) -> Dict[str, dict]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}



def _safe_json_save(path: str, payload: Dict[str, dict]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass



def _profile_key() -> str:
    w_px, h_px = get_screen_pixels()
    return f"{platform.system()}:{w_px}x{h_px}"


# -----------------------------------------------------------------------------
# Monitor calibration
# -----------------------------------------------------------------------------


def _estimate_arcmin_per_pixel(screen_width_cm: float, viewing_distance_cm: float, res_x: int) -> float:
    theta_deg = math.degrees(math.atan((screen_width_cm / 2.0) / viewing_distance_cm))
    return theta_deg * 60.0 / (res_x / 2.0)


def load_or_calibrate_monitor() -> Optional[Tuple[monitors.Monitor, MonitorGeometry]]:
    """Load cached monitor geometry or ask the user to measure screen width manually.

    This intentionally keeps the simpler and more explicit workflow:
    the user measures the physical screen width in centimeters and enters it directly.
    No visual object-based auto-calibration is attempted.
    """
    ident = _profile_key()
    profiles = _safe_json_load(MONITOR_CACHE_FILE)
    profile = profiles.get(ident)

    w_px, h_px = get_screen_pixels()

    if profile is None:
        dlg_dict = {
            "Screen width (cm)": SCREEN_WIDTH_CM_DEFAULT,
            "Viewing distance (cm)": VIEWING_DISTANCE_CM_DEFAULT,
        }
        dlg = gui.DlgFromDict(
            dlg_dict,
            title="Screen calibration",
            order=["Screen width (cm)", "Viewing distance (cm)"],
        )
        if not dlg.OK:
            return None

        try:
            width_cm = float(dlg_dict["Screen width (cm)"])
            distance_cm = float(dlg_dict["Viewing distance (cm)"])
        except Exception:
            width_cm = float(SCREEN_WIDTH_CM_DEFAULT)
            distance_cm = float(VIEWING_DISTANCE_CM_DEFAULT)

        profile = {
            "width_cm": width_cm,
            "distance_cm": distance_cm,
            "size_pix": [w_px, h_px],
        }
        profiles[ident] = profile
        _safe_json_save(MONITOR_CACHE_FILE, profiles)

    mon = monitors.Monitor(ident)
    mon.setWidth(float(profile["width_cm"]))
    mon.setDistance(float(profile["distance_cm"]))
    mon.setSizePix([int(profile["size_pix"][0]), int(profile["size_pix"][1])])

    geom = MonitorGeometry(
        width_cm=float(profile["width_cm"]),
        distance_cm=float(profile["distance_cm"]),
        res_x=int(profile["size_pix"][0]),
        res_y=int(profile["size_pix"][1]),
        arcmin_per_pix=_estimate_arcmin_per_pixel(
            screen_width_cm=float(profile["width_cm"]),
            viewing_distance_cm=float(profile["distance_cm"]),
            res_x=int(profile["size_pix"][0]),
        ),
    )
    return mon, geom


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------


def _ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        fallback = os.path.join(tempfile.gettempdir(), "psychopy_data")
        os.makedirs(fallback, exist_ok=True)
        return fallback


# -----------------------------------------------------------------------------
# Cue drawing
# -----------------------------------------------------------------------------


def build_informative_precue_vertices(angle_set: int, direction: int, angle_deviation_deg: float, geom: MonitorGeometry) -> List[Tuple[float, float]]:
    """Build the central triangular informative pre-cue used by the previous working implementation.

    angle_set = 0: horizontal-axis task -> cue indicates leftward vs rightward base direction
    angle_set = 1: vertical-axis task   -> cue indicates downward vs upward base direction

    The cue is drawn at fixation. Its depth is fixed at about 1 degree, while its width
    encodes relative trial difficulty: a smaller wedge means a smaller angle deviation
    and therefore a harder trial.
    """
    depth_deg = 1.0
    half_span_deg = max(float(angle_deviation_deg), MIN_CUE_SPAN_DEG) / 60.0

    depth_pix = deg_to_pix(depth_deg, geom)
    half_span_pix = max(2.0, deg_to_pix(half_span_deg, geom))

    if angle_set == 0:
        if direction == 1:  # rightward base direction
            return [(0.0, 0.0), (depth_pix, -half_span_pix), (depth_pix, half_span_pix)]
        return [(0.0, 0.0), (-depth_pix, -half_span_pix), (-depth_pix, half_span_pix)]

    if direction == 1:  # downward base direction
        return [(0.0, 0.0), (-half_span_pix, -depth_pix), (half_span_pix, -depth_pix)]
    return [(0.0, 0.0), (-half_span_pix, depth_pix), (half_span_pix, depth_pix)]


def draw_triangular_precue(
    win: visual.Window,
    geom: MonitorGeometry,
    angle_set: int,
    direction: int,
    angle_deviation_deg: float,
    color: Sequence[float] = (1, 1, 1),
) -> None:
    vertices = build_informative_precue_vertices(angle_set, direction, angle_deviation_deg, geom)
    cue = visual.ShapeStim(
        win,
        vertices=vertices,
        fillColor=color,
        lineColor=color,
        lineWidth=1,
        closeShape=True,
        units="pix",
    )
    cue.draw()


# -----------------------------------------------------------------------------
# Core session implementation
# -----------------------------------------------------------------------------


def _build_fixed_stair_sequence(total_trials: int) -> List[int]:
    """Return a shuffled list of staircase IDs with exactly three staircases.

    IDs are 1, 2, 3 for direct continuity with the original MATLAB logic.
    """
    base = total_trials // N_STAIRCASES
    rem = total_trials % N_STAIRCASES
    seq: List[int] = []
    for staircase_id in range(1, N_STAIRCASES + 1):
        reps = base + (1 if staircase_id <= rem else 0)
        seq.extend([staircase_id] * reps)
    np.random.shuffle(seq)
    return seq


@dataclass
class TrialResult:
    trial: int
    angle_dev_deg: float
    staircase: int
    direction_code: int
    orientation_sign: int
    rt_s: float
    correct: int
    angle_deg: float



def run_fba_tilt_global_session(
    win: visual.Window,
    geom: MonitorGeometry,
    subject_id: str,
    location_deg_internal: Tuple[float, float],
    angle_set: int,
    total_trials: int,
    use_precue: bool,
    save_dir: str,
) -> Dict[str, object]:
    """Run one home-training session.

    Parameters
    ----------
    location_deg_internal:
        (H, V_internal) in visual-field sign conventions after the internal Y inversion
        used by this script. This keeps saved metadata explicit and reversible.
    angle_set:
        0 = horizontal base directions, respond UP/DOWN
        1 = vertical base directions, respond LEFT/RIGHT
    """
    save_dir = _ensure_dir(save_dir)

    # Fixed protocol settings.
    h_ecc_stim_deg, v_ecc_stim_internal_deg = location_deg_internal
    fixation_duration = FIXATION_DURATION_S
    cue_duration = CUE_DURATION_S
    post_cue_isi = POST_CUE_ISI_S
    cue_color = (1, 1, 1)

    # Probe frame timing from the actual display. We fall back gracefully if PsychoPy
    # cannot estimate it.
    refresh = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=20)
    if refresh is None or refresh <= 0:
        refresh = 60.0
    frame_ms = 1000.0 / refresh
    mv_length = int(round(STIMULUS_DURATION_MS / frame_ms))
    lifetime_frames = max(1, int(round(DOT_LIFETIME_MS / frame_ms)))

    # Convert the stimulus center to pixels for the actual RDK drawing code.
    stim_x_pix = deg_to_pix(h_ecc_stim_deg, geom)
    stim_y_pix = -deg_to_pix(v_ecc_stim_internal_deg, geom)
    aperture_radius_pix = deg_to_pix(APERTURE_RADIUS_DEG, geom)
    dot_step_pix = deg_to_pix(DOT_SPEED_DEG_PER_S / refresh, geom)
    dot_size_pix = max(2, int(math.floor(DOT_SIZE_ARCMIN / geom.arcmin_per_pix)))

    # Number of dots is defined by physical density in the aperture area.
    area_deg2 = math.pi * (APERTURE_RADIUS_DEG ** 2)
    n_dots = int(round(DOT_DENSITY_PER_DEG2 * area_deg2))

    # PsychoPy stimuli reused across trials.
    fixation = visual.Circle(win, radius=FIX_OUTER_RADIUS_DEG, fillColor=-1, lineColor=-1, pos=(0, 0), units="deg")
    fixation_inner = visual.Circle(win, radius=FIX_INNER_RADIUS_DEG, fillColor=1, lineColor=1, pos=(0, 0), units="deg")
    dots = visual.ElementArrayStim(
        win,
        nElements=n_dots,
        elementTex=None,
        elementMask="circle",
        xys=np.zeros((n_dots, 2)),
        sizes=[dot_size_pix] * n_dots,
        units="pix",
        colors=[DOT_COLOR_RGB] * n_dots,
        colorSpace="rgb",
        sfs=0,
    )

    # Fixed 3-staircase protocol.
    stair_idxs = INITIAL_STAIR_IDXS.copy()
    streak_counts = [0, 0, 0]
    stair_sequence = _build_fixed_stair_sequence(total_trials)

    # Audio feedback. If audio initialization fails, we continue silently.
    try:
        snd_start = sound.Sound(value=1000, secs=0.05)
        snd_correct = sound.Sound(value=1200, secs=0.12)
        snd_incorrect = sound.Sound(value=800, secs=0.12)
    except Exception:
        snd_start = snd_correct = snd_incorrect = None

    results: List[TrialResult] = []
    rt_clock = core.Clock()
    win.color = BACKGROUND_RGB
    win.flip()

    aborted = False
    error_info: Optional[str] = None

    try:
        for trial_index, which_stair in enumerate(stair_sequence, start=1):
            angle_dev_deg = ANGLE_RANGE_DEG[stair_idxs[which_stair - 1]]

            # In continuity with the prior code branch:
            # angle_set=0 uses horizontal base directions, queried with UP/DOWN
            # angle_set=1 uses vertical base directions, queried with LEFT/RIGHT
            direction = int(np.random.randint(1, 3))
            orientation = int(np.random.choice([-1, 1]))

            if angle_set == 0 and direction == 1:
                angle_deg = 0.0 + angle_dev_deg * orientation
            elif angle_set == 0 and direction == 2:
                angle_deg = 180.0 + angle_dev_deg * orientation
            elif angle_set == 1 and direction == 1:
                angle_deg = 270.0 + angle_dev_deg * orientation
            else:
                angle_deg = 90.0 + angle_dev_deg * orientation

            angle_rad = math.radians(angle_deg)
            vx = dot_step_pix * math.cos(angle_rad)
            vy = dot_step_pix * math.sin(angle_rad)

            # Response mapping is intentionally explicit rather than abstract.
            if angle_set == 0:
                correct_key = "up" if vy > 0 else "down"
                incorrect_key = "down" if correct_key == "up" else "up"
            else:
                correct_key = "left" if vx < 0 else "right"
                incorrect_key = "right" if correct_key == "left" else "left"

            # Fixation period.
            win.callOnFlip(event.clearEvents, eventType="keyboard")
            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(fixation_duration)

            # Pre-cue. This uses the triangular informative pre-cue from the previous working implementation.
            if use_precue:
                draw_triangular_precue(
                    win=win,
                    geom=geom,
                    angle_set=angle_set,
                    direction=direction,
                    angle_deviation_deg=angle_dev_deg,
                    color=cue_color,
                )
                fixation.draw()
                fixation_inner.draw()
                win.flip()
                core.wait(cue_duration)

            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(post_cue_isi)

            # Random initial dot positions and ages.
            positions = np.zeros((n_dots, 2), dtype=float)
            ages = np.zeros(n_dots, dtype=int)
            for i in range(n_dots):
                while True:
                    x = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                    y = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                    if x * x + y * y <= aperture_radius_pix * aperture_radius_pix:
                        positions[i, 0] = x
                        positions[i, 1] = y
                        ages[i] = int(np.random.randint(1, lifetime_frames + 1))
                        break

            if snd_start is not None:
                snd_start.play()
                core.wait(0.05)

            win.callOnFlip(event.clearEvents, eventType="keyboard")
            win.flip()
            rt_clock.reset()

            # Tilt-global RDK: all dots share the same motion vector on a given trial.
            for _ in range(mv_length):
                for i in range(n_dots):
                    x = positions[i, 0] + vx
                    y = positions[i, 1] + vy
                    age = ages[i] + 1

                    if age > lifetime_frames:
                        while True:
                            rx = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                            ry = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                            if rx * rx + ry * ry <= aperture_radius_pix * aperture_radius_pix:
                                x, y = rx, ry
                                age = 1
                                break

                    # Wrap and clip back into the aperture.
                    if x > aperture_radius_pix:
                        x -= 2.0 * aperture_radius_pix
                    elif x < -aperture_radius_pix:
                        x += 2.0 * aperture_radius_pix
                    if y > aperture_radius_pix:
                        y -= 2.0 * aperture_radius_pix
                    elif y < -aperture_radius_pix:
                        y += 2.0 * aperture_radius_pix
                    if x * x + y * y > aperture_radius_pix * aperture_radius_pix:
                        while True:
                            rx = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                            ry = (np.random.rand() - 0.5) * 2.0 * aperture_radius_pix
                            if rx * rx + ry * ry <= aperture_radius_pix * aperture_radius_pix:
                                x, y = rx, ry
                                break

                    positions[i, 0] = x
                    positions[i, 1] = y
                    ages[i] = age

                xys = positions.copy()
                xys[:, 0] += stim_x_pix
                xys[:, 1] += stim_y_pix
                dots.xys = xys
                dots.draw()
                fixation.draw()
                fixation_inner.draw()
                win.flip()

            fixation.draw()
            fixation_inner.draw()
            win.flip()

            rt_s = float("nan")
            correct = 0
            keys = event.waitKeys(keyList=[correct_key, incorrect_key, "escape"], timeStamped=rt_clock)
            if keys:
                key, rt_s = keys[0]
                if key == "escape":
                    aborted = True
                    break
                if key == correct_key:
                    correct = 1
                    if snd_correct is not None:
                        snd_correct.play()
                else:
                    correct = 0
                    if snd_incorrect is not None:
                        snd_incorrect.play()

            # 3-down / 1-up bookkeeping as in the branch we are mirroring.
            idx = which_stair - 1
            if correct:
                streak_counts[idx] += 1
                if streak_counts[idx] >= 3:
                    stair_idxs[idx] = min(stair_idxs[idx] + 1, len(ANGLE_RANGE_DEG) - 1)
                    streak_counts[idx] = 0
            else:
                stair_idxs[idx] = max(stair_idxs[idx] - 1, 0)
                streak_counts[idx] = 0

            results.append(
                TrialResult(
                    trial=trial_index,
                    angle_dev_deg=float(angle_dev_deg),
                    staircase=which_stair,
                    direction_code=direction,
                    orientation_sign=orientation,
                    rt_s=float(rt_s),
                    correct=int(correct),
                    angle_deg=float(angle_deg),
                )
            )

            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(0.5)

    except Exception as exc:
        error_info = repr(exc)
        traceback.print_exc()

    # Deterministic save block: always attempt to save whatever happened.
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{subject_id}_FBA_TILTGLOBAL_{ts}"
    csv_path = os.path.join(save_dir, base + "_trials.csv")
    summary_path = os.path.join(save_dir, base + "_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "angle_dev_deg",
            "staircase",
            "direction_code",
            "orientation_sign",
            "rt_s",
            "correct",
            "angle_deg",
        ])
        for row in results:
            writer.writerow([
                row.trial,
                row.angle_dev_deg,
                row.staircase,
                row.direction_code,
                row.orientation_sign,
                row.rt_s,
                row.correct,
                row.angle_deg,
            ])

    accuracy = (100.0 * sum(r.correct for r in results) / max(1, len(results))) if results else float("nan")
    staircase_estimate = float(np.mean([ANGLE_RANGE_DEG[idx] for idx in stair_idxs]))

    summary: Dict[str, object] = {
        "subject": subject_id,
        "task": "FBA_TiltGlobal_HomeTraining",
        "source_lineage": {
            "training": "TrainingCodes",
            "analysis": "DataFitting-compatible",
        },
        "n_trials_completed": len(results),
        "accuracy_percent": accuracy,
        "coarse_session_staircase_estimate_deg": staircase_estimate,
        "location_deg_internal": {
            "H": float(h_ecc_stim_deg),
            "V_internal": float(v_ecc_stim_internal_deg),
        },
        "angle_set": int(angle_set),
        "use_precue": bool(use_precue),
        "fixed_protocol": {
            "n_staircases": N_STAIRCASES,
            "angle_range_deg": ANGLE_RANGE_DEG,
            "dot_density_per_deg2": DOT_DENSITY_PER_DEG2,
            "dot_speed_deg_per_s": DOT_SPEED_DEG_PER_S,
            "aperture_radius_deg": APERTURE_RADIUS_DEG,
            "stimulus_duration_ms": STIMULUS_DURATION_MS,
            "dot_lifetime_ms": DOT_LIFETIME_MS,
            "cue_duration_s": CUE_DURATION_S,
            "post_cue_isi_s": POST_CUE_ISI_S,
        },
        "monitor_geometry": {
            "screen_width_cm": geom.width_cm,
            "viewing_distance_cm": geom.distance_cm,
            "resolution_px": [geom.res_x, geom.res_y],
            "arcmin_per_pix": geom.arcmin_per_pix,
        },
        "aborted": aborted,
        "error": error_info,
        "timestamp": ts,
        "trial_csv": os.path.basename(csv_path),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> Optional[Dict[str, object]]:
    info = {
        "Subject ID": "Thomas",
        "Angle set (0=Horizontal UP/DOWN, 1=Vertical LEFT/RIGHT)": 0,
        "Stimulus horizontal eccentricity (deg; Left(-) to Right(+))": "0.0",
        "Stimulus vertical eccentricity (deg; Down(-) to Up(+))": "0.0",
        "Number of Trials": TOTAL_TRIALS_DEFAULT,
        "Show pre-cue?": True,
    }
    dlg = gui.DlgFromDict(
        info,
        title="Huxlin-style FBA Tilt Global training",
        order=[
            "Subject ID",
            "Angle set (0=Horizontal UP/DOWN, 1=Vertical LEFT/RIGHT)",
            "Stimulus horizontal eccentricity (deg; Left(-) to Right(+))",
            "Stimulus vertical eccentricity (deg; Down(-) to Up(+))",
            "Number of Trials",
            "Show pre-cue?",
        ],
    )
    if not dlg.OK:
        return None

    try:
        subject_id = str(info["Subject ID"])
        total_trials = int(info["Number of Trials"])
        angle_set = int(info["Angle set (0=Horizontal UP/DOWN, 1=Vertical LEFT/RIGHT)"])
        use_precue = bool(info["Show pre-cue?"])
        h_ecc_field = float(info["Stimulus horizontal eccentricity (deg; Left(-) to Right(+))"])
        v_ecc_field = float(info["Stimulus vertical eccentricity (deg; Down(-) to Up(+))"])
    except Exception:
        err = gui.Dlg(title="Input error")
        err.addText("One or more inputs were invalid.")
        err.show()
        return None

    mon_geom = load_or_calibrate_monitor()
    if mon_geom is None:
        return None
    mon, geom = mon_geom

    # Internal Y convention: PsychoPy positive Y is up, whereas we want a direct and
    # explicit mapping from user-entered visual-field coordinates. We therefore store
    # both the entered field coordinates and the internally flipped coordinates.
    h_internal = h_ecc_field
    v_internal = -v_ecc_field

    win = visual.Window(
        size=(geom.res_x, geom.res_y),
        fullscr=True,
        monitor=mon,
        units="deg",
        color=BACKGROUND_RGB,
        colorSpace="rgb",
        allowGUI=False,
    )

    response_text = (
        "Use the UP and DOWN arrow keys.\n\nUP = motion tilted upward\nDOWN = motion tilted downward"
        if angle_set == 0
        else "Use the LEFT and RIGHT arrow keys.\n\nLEFT = motion tilted leftward\nRIGHT = motion tilted rightward"
    )

    readme = (
        "READ ME / SETUP\n\n"
        "This program implements a Huxlin-style FDD training task using a coherent-motion\n"
        "random-dot stimulus and a triangular central informative pre-cue.\n\n"
        "Important: this is a home-training implementation. It does NOT perform eye tracking.\n"
        "Keep your eyes strictly on the central fixation dot.\n\n"
        f"Stimulus location in visual field (deg): H = {h_ecc_field:.2f}, V = {v_ecc_field:.2f}\n"
        f"Viewing distance: {geom.distance_cm:.2f} cm\n"
        f"Screen width: {geom.width_cm:.2f} cm\n"
        f"Difficulty scale (deg): {ANGLE_RANGE_DEG}\n"
        f"Dot density: {DOT_DENSITY_PER_DEG2} dots/deg²\n"
        f"Dot speed: {DOT_SPEED_DEG_PER_S} deg/s\n\n"
        + response_text
        + f"\n\nPre-cue enabled: {'Yes' if use_precue else 'No'}"
        + "\n\nPress SPACE to start or ESC to quit."
    )

    instr = visual.TextStim(win, text=readme, color=-1, height=0.55, wrapWidth=24)
    instr.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        win.close()
        return None

    summary = run_fba_tilt_global_session(
        win=win,
        geom=geom,
        subject_id=subject_id,
        location_deg_internal=(h_internal, v_internal),
        angle_set=angle_set,
        total_trials=total_trials,
        use_precue=use_precue,
        save_dir=os.path.join(BASE_DIR, DATA_DIRNAME),
    )

    if summary.get("error"):
        msg = (
            "Session ended due to an unexpected error.\n\n"
            f"Trials saved: {summary.get('n_trials_completed', 0)}\n"
            "Please review the summary JSON and trial CSV.\n\n"
            "Press any key to exit."
        )
    elif summary.get("aborted"):
        msg = (
            "Session interrupted.\n\n"
            f"Trials saved: {summary.get('n_trials_completed', 0)}\n"
            f"Accuracy so far: {summary.get('accuracy_percent', float('nan')):.1f}%\n"
            "Press any key to exit."
        )
    else:
        msg = (
            "Training complete.\n\n"
            f"Accuracy: {summary.get('accuracy_percent', float('nan')):.1f}%\n"
            f"Coarse session staircase estimate: {summary.get('coarse_session_staircase_estimate_deg', float('nan')):.2f} deg\n"
            "\nUse the standalone analysis script for the Huxlin-style Weibull fit.\n\n"
            "Press any key to exit."
        )

    visual.TextStim(win, text=msg, color=-1, height=0.8, wrapWidth=22).draw()
    win.flip()
    event.waitKeys()
    win.close()
    return summary


if __name__ == "__main__":
    main()
