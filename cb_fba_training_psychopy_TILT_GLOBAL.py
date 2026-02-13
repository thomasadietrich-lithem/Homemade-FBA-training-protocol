# cb_fba_training_psychopy_DR.py
# # cb_fba_training_psychopy_TILT_GLOBAL.py
# PsychoPy implementation of Huxlin Lab FBA motion training ("tilt global" RDK)
#
# In this version, ALL dots move in the same direction on each trial (no direction dispersion).
# The participant reports the global motion tilt:
#   - Horizontal axis training: UP vs DOWN (tilt above/below horizontal)
#   - Vertical axis training: LEFT vs RIGHT (tilt left/right of vertical)
#
#   - Per-dot motion directions (vx_i, vy_i) instead of a single (vx, vy).
#   - When a dot respawns (lifetime reset), it gets a new random direction
#     from the same distribution.
#   - Response mapping is intuitive:
#       angle_set = 0 (horizontal axis): UP = tilt up, DOWN = tilt down
#       angle_set = 1 (vertical axis): LEFT = tilt left, RIGHT = tilt right

from psychopy import visual, core, event, gui, monitors, sound
import numpy as np
import os, csv, math, json, datetime, traceback, tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Monitor / geometry helpers ----------

def get_screen_pixels():
    try:
        import pyglet
        d = pyglet.canvas.get_display()
        s = d.get_default_screen()
        return int(s.width), int(s.height)
    except Exception:
        return 1920, 1080

def load_or_ask_monitor():
    import platform
    w_px, h_px = get_screen_pixels()
    ident = f"{platform.system()}:{w_px}x{h_px}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(base_dir, "monitor_profiles.json")
    cache = {}
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    prof = cache.get(ident)

    if prof is None:
        dlg_dict = {
            "Screen width (cm)": 61.0,
            "Viewing distance (cm)": 42.0,
        }
        dlg = gui.DlgFromDict(dlg_dict, title="Screen calibration",
                              order=["Screen width (cm)", "Viewing distance (cm)"])
        if not dlg.OK:
            return None
        try:
            width_cm = float(dlg_dict["Screen width (cm)"])
            dist_cm = float(dlg_dict["Viewing distance (cm)"])
        except Exception:
            width_cm, dist_cm = 61.0, 42.0

        prof = {
            "width_cm": width_cm,
            "distance_cm": dist_cm,
            "size_pix": [w_px, h_px],
        }
        cache[ident] = prof
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    mon = monitors.Monitor(ident)
    mon.setWidth(prof["width_cm"])
    mon.setDistance(prof["distance_cm"])
    mon.setSizePix(prof["size_pix"])

    screen_width_cm = prof["width_cm"]
    viewing_dist_cm = prof["distance_cm"]
    res_x = prof["size_pix"][0]

    theta_deg = math.degrees(math.atan((screen_width_cm/2.0)/viewing_dist_cm))
    arcmin_per_pix = theta_deg * 60.0 / (res_x/2.0)

    geom = {
        "screen_width_cm": screen_width_cm,
        "viewing_dist_cm": viewing_dist_cm,
        "res_x": res_x,
        "res_y": prof["size_pix"][1],
        "arcmin_per_pix": arcmin_per_pix,
    }
    return mon, geom

def deg_to_pix(deg, geom):
    arcmin = deg * 60.0
    return arcmin / geom["arcmin_per_pix"]


# ---------- SAFE SAVE helpers (PATCH 2026 - infrastructure only) ----------
def _resolve_save_dir(save_dir: str) -> str:
    if save_dir is None or str(save_dir).strip() == "":
        save_dir = "data"
    save_dir = str(save_dir)
    if os.path.isabs(save_dir):
        return save_dir
    return os.path.join(BASE_DIR, save_dir)

def _safe_makedirs(path_):
    try:
        os.makedirs(path_, exist_ok=True)
        return True, None
    except Exception as e:
        return False, repr(e)

def _emergency_dir():
    for cand in [
        os.path.join(os.path.expanduser("~"), "psychopy_data"),
        os.path.join(tempfile.gettempdir(), "psychopy_data"),
    ]:
        ok, _ = _safe_makedirs(cand)
        if ok:
            return cand
    return tempfile.gettempdir()

# ---------- Core FBA RDK training (Tilt Global) ----------

def run_fba_rdk_tilt_global(win, geom, subject_id, location_deg_internal, angle_set=0,
                                n_staircases=3, total_trials=300,
                                save_dir="data"):

    # PATCH 2026 (infrastructure only): make save path deterministic & writable
    resolved_save_dir = _resolve_save_dir(save_dir)

    H_ecc_fix = 0.0
    V_ecc_fix = 0.0
    H_ecc_stim, V_ecc_stim_internal = location_deg_internal

    cue_duration = 0.2
    cue_color = [1, 1, 1]

    stimulus_duration_ms = 500.0
    aperture_radius_deg = 2.5
    dot_density = 3.5
    initial_dot_size_arcmin = 14.0
    dot_color = -1.0
    dot_speed_deg_per_s = 10.0
    dot_lifetime_ms = 200.0

    background = 0.5

    angle_range = [85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]
    stair1 = 1
    stair2 = 4
    stair3 = 8

    total_trials = int(total_trials)

    refresh = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=20)
    if refresh is None or refresh <= 0:
        refresh = 60.0
    mv_length = int(round(stimulus_duration_ms / (1000.0/refresh)))
    lifetime_frames = int(round(dot_lifetime_ms / (1000.0/refresh)))

    fix_x_deg = H_ecc_fix
    fix_y_deg = V_ecc_fix
    stim_x_deg = H_ecc_stim

    dot_size_pix = int(math.floor(initial_dot_size_arcmin / geom["arcmin_per_pix"]))
    if dot_size_pix < 2:
        dot_size_pix = 2

    area_deg2 = math.pi * (aperture_radius_deg ** 2)
    n_dots = int(round(dot_density * area_deg2))

    dot_step_deg = dot_speed_deg_per_s / refresh

    fixation = visual.Circle(win, radius=0.1, fillColor=-1, lineColor=-1,
                             pos=(fix_x_deg, fix_y_deg), units="deg")
    fixation_inner = visual.Circle(win, radius=0.05, fillColor=1, lineColor=1,
                                   pos=(fix_x_deg, fix_y_deg), units="deg")

    stim_x_pix = deg_to_pix(stim_x_deg, geom)
    stim_y_pix = -deg_to_pix(V_ecc_stim_internal, geom)

    dots = visual.ElementArrayStim(
        win,
        nElements=n_dots,
        elementTex=None,
        elementMask="circle",
        xys=np.zeros((n_dots, 2)),
        sizes=[dot_size_pix] * n_dots,
        units="pix",
        colors=[dot_color] * n_dots,
        colorSpace="rgb",
        sfs=0
    )

    stair_array = []
    # Build a roughly balanced, randomized list of staircase IDs for the requested total_trials
    base = total_trials // n_staircases
    rem = total_trials % n_staircases
    for s in range(n_staircases):
        reps = base + (1 if s < rem else 0)
        stair_array.extend([s+1] * reps)
    np.random.shuffle(stair_array)

    staircount1 = staircount2 = staircount3 = 0

    results = []
    trial = 0
    clock = core.Clock()

    snd_start = sound.Sound(value=1000, secs=0.05)
    snd_correct = sound.Sound(value=1200, secs=0.12)
    snd_incorrect = sound.Sound(value=800, secs=0.12)

    win.color = background
    win.flip()

    abort_requested = False  # PATCH SAFE EXIT: infrastructure only
    error_info = None  # PATCH SAFE EXIT: capture unexpected errors
    summary_out = None  # PATCH SAFE EXIT

    try:
        while trial < total_trials:
            trial += 1
            which_stair = stair_array[trial-1]
            if which_stair == 1:
                angle_deviationP = angle_range[stair1-1]
            elif which_stair == 2:
                angle_deviationP = angle_range[stair2-1]
            else:
                angle_deviationP = angle_range[stair3-1]
    
            direction = np.random.randint(1, 3)  # 1 or 2
            orientation = np.random.choice([-1, 1])
    
            # ================== BASE ANGLE (central direction) ==================
            if angle_set == 0 and direction == 1:
                angle_deg = 0 + angle_deviationP * orientation
            elif angle_set == 0 and direction == 2:
                angle_deg = 180 + angle_deviationP * orientation
            elif angle_set == 1 and direction == 1:
                angle_deg = 270 + angle_deviationP * orientation
            else:
                angle_deg = 90 + angle_deviationP * orientation
    
            # Intuitive mapping based on central direction:
            # horizontal axis -> use vertical component
            # vertical axis   -> use horizontal component
            step_pix = deg_to_pix(dot_step_deg, geom)
            angle_rad_central = math.radians(angle_deg)
            vx_central = step_pix * math.cos(angle_rad_central)
            vy_central = step_pix * math.sin(angle_rad_central)
    
            if angle_set == 0:
                # Horizontal axis, decide UP vs DOWN
                if vy_central > 0:
                    correct_key = "up"
                    incorrect_key = "down"
                else:
                    correct_key = "down"
                    incorrect_key = "up"
            else:
                # Vertical axis, decide LEFT vs RIGHT
                if vx_central < 0:
                    correct_key = "left"
                    incorrect_key = "right"
                else:
                    correct_key = "right"
                    incorrect_key = "left"
    
            # ================== PRE-CUE ==================
            win.callOnFlip(event.clearEvents, eventType="keyboard")
            win.flip()
            fx_pix = deg_to_pix(fix_x_deg, geom)
            fy_pix = -deg_to_pix(fix_y_deg, geom)
            target_x_pix = stim_x_pix
            target_y_pix = stim_y_pix
    
            cue_stim = visual.ShapeStim(
                win,
                vertices=[(fx_pix, fy_pix), (target_x_pix, target_y_pix)],
                lineColor=cue_color,
                lineWidth=2,
                units="pix"
            )
            cue_stim.draw()
            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(cue_duration)
    
            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(0.05)
    
            # ================== INITIALIZE DOT POSITIONS + DIRECTIONS ==================
            stimulus_radius_pix = deg_to_pix(aperture_radius_deg, geom)
            positions = np.zeros((n_dots, 2), dtype=float)
            ages = np.zeros(n_dots, dtype=int)
    
            # Random initial positions
            for i in range(n_dots):
                while True:
                    x = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                    y = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                    if x*x + y*y <= stimulus_radius_pix**2:
                        positions[i, 0] = x
                        positions[i, 1] = y
                        ages[i] = np.random.randint(1, lifetime_frames+1)
                        break
    
            # GLOBAL DIRECTION (Tilt Global): all dots share the same motion direction this trial
            vx_dots = np.full(n_dots, vx_central, dtype=float)
            vy_dots = np.full(n_dots, vy_central, dtype=float)
    
            # ================== START BEEP ==================
            snd_start.play()
            core.wait(0.05)
    
            # ================== PLAY RDK ==================
            win.callOnFlip(event.clearEvents, eventType="keyboard")
            win.flip()
            clock.reset()
    
            for f in range(mv_length):
                # Update positions
                for i in range(n_dots):
                    x = positions[i, 0] + vx_dots[i]
                    y = positions[i, 1] + vy_dots[i]
                    age = ages[i] + 1
    
                    # Lifetime reset
                    if age > lifetime_frames:
                        while True:
                            rx = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                            ry = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                            if rx*rx + ry*ry <= stimulus_radius_pix**2:
                                x, y = rx, ry
                                age = 1
                                break
    
                    # Wrap around
                    if x > stimulus_radius_pix:
                        x -= 2*stimulus_radius_pix
                    elif x < -stimulus_radius_pix:
                        x += 2*stimulus_radius_pix
                    if y > stimulus_radius_pix:
                        y -= 2*stimulus_radius_pix
                    elif y < -stimulus_radius_pix:
                        y += 2*stimulus_radius_pix
    
                    # Keep inside circle
                    if x*x + y*y > stimulus_radius_pix**2:
                        while True:
                            rx = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                            ry = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                            if rx*rx + ry*ry <= stimulus_radius_pix**2:
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
    
            # ================== RESPONSE ==================
            rt = None
            correct = 0
            keys = event.waitKeys(maxWait=2.0,
                                  keyList=[correct_key, incorrect_key, "escape"],
                                  timeStamped=clock)
            if keys:
                key, rt = keys[0]
                if key == "escape":
                    abort_requested = True  # PATCH SAFE EXIT: do not hard-quit
                    break
                if key == correct_key:
                    correct = 1
                    snd_correct.play()
                else:
                    correct = 0
                    snd_incorrect.play()
    
            # ================== STAIRCASE UPDATE ==================
            if correct:
                if which_stair == 1:
                    staircount1 += 1
                    if staircount1 >= 3:
                        stair1 = min(stair1 + 1, len(angle_range))
                        staircount1 = 0
                elif which_stair == 2:
                    staircount2 += 1
                    if staircount2 >= 3:
                        stair2 = min(stair2 + 1, len(angle_range))
                        staircount2 = 0
                else:
                    staircount3 += 1
                    if staircount3 >= 3:
                        stair3 = min(stair3 + 1, len(angle_range))
                        staircount3 = 0
            else:
                if which_stair == 1:
                    stair1 = max(1, stair1 - 1)
                    staircount1 = 0
                elif which_stair == 2:
                    stair2 = max(1, stair2 - 1)
                    staircount2 = 0
                else:
                    stair3 = max(1, stair3 - 1)
                    staircount3 = 0
    
            results.append([
                trial,
                angle_deviationP,
                which_stair,
                direction,
                orientation,
                rt if rt is not None else float("nan"),
                correct,
                angle_deg
            ])
    
            fixation.draw()
            fixation_inner.draw()
            win.flip()
            core.wait(0.5)
    
    except Exception as e:
        error_info = repr(e)

    finally:
        # ================== SUMMARY / SAVE ==================
        # PATCH 2026 (infrastructure only): attempt to save outputs even if session aborted or an exception occurred.
        try:
            correct_trials = sum(r[6] for r in results)
            accuracy = 100.0 * correct_trials / max(1, len(results))
            final_thresh = (angle_range[stair1-1] +
                            angle_range[stair2-1] +
                            angle_range[stair3-1]) / 3.0

            ok, mk_err = _safe_makedirs(resolved_save_dir)
            if not ok:
                resolved_save_dir = _emergency_dir()

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{subject_id}_FBA_TILT_{ts}"

            csv_path = os.path.join(resolved_save_dir, base + "_trials.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["trial", "angle_dev_deg", "staircase", "direction_code",
                            "orientation_sign", "rt_s", "correct", "angle_deg"])
                for row in results:
                    w.writerow(row)

            summary_path = os.path.join(resolved_save_dir, base + "_summary.json")
            summary = {
                "subject": subject_id,
                "task": "FBA_RDK_TiltGlobal",
                "location_deg_internal": {"H": H_ecc_stim, "V_internal": V_ecc_stim_internal},
                "angle_set": angle_set,
                "n_trials": len(results),
                "accuracy_percent": accuracy,
                "final_threshold_deg": final_thresh,
                "stair_levels": {"stair1": stair1, "stair2": stair2, "stair3": stair3},
                "angle_range": angle_range,
                "timestamp": ts,
                "aborted": bool(abort_requested),
                "error": error_info,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            summary_out = summary

        except Exception as save_exc:
            fallback_dir = _emergency_dir()
            ts2 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base2 = f"{subject_id}_FBA_TILT_RECOVERY_{ts2}"

            try:
                rec_csv = os.path.join(fallback_dir, base2 + "_trials.csv")
                with open(rec_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["trial", "angle_dev_deg", "staircase", "direction_code",
                                "orientation_sign", "rt_s", "correct", "angle_deg"])
                    for row in results:
                        w.writerow(row)
            except Exception:
                pass

            try:
                rec_json = os.path.join(fallback_dir, base2 + "_recovery.json")
                payload = {
                    "subject": subject_id,
                    "n_trials": len(results),
                    "aborted": bool(abort_requested),
                    "error": error_info,
                    "save_error": repr(save_exc),
                }
                with open(rec_json, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
            except Exception:
                pass

    return summary_out



# Backward-compatible alias
run_fba_rdk_direction_range = run_fba_rdk_tilt_global

# ---------- Main entry point ----------

def main():
    info = {
        "Subject ID": "Thomas",
        "Angle set (0=Horizontal: UP/DOWN, 1=Vertical: LEFT/RIGHT)": 1,
        "Stimulus horizontal eccentricity (deg; Left(-) to Right(+))": -3.0,
        "Stimulus vertical eccentricity (deg; Down(-) to Up(+))": -5.0,
        "Number of Trials": 300,
    }
    dlg = gui.DlgFromDict(info, title="CB training (Tilt Global, FBA)", order=["Subject ID","Angle set (0=Horizontal: UP/DOWN, 1=Vertical: LEFT/RIGHT)","Stimulus horizontal eccentricity (deg; Left(-) to Right(+))","Stimulus vertical eccentricity (deg; Down(-) to Up(+))","Number of Trials"])
    if not dlg.OK:
        return None

    subject_id = str(info["Subject ID"])
    total_trials = int(info["Number of Trials"])
    angle_set = int(info["Angle set (0=Horizontal: UP/DOWN, 1=Vertical: LEFT/RIGHT)"])
    H_ecc_field = float(info["Stimulus horizontal eccentricity (deg; Left(-) to Right(+))"])
    V_ecc_field = float(info["Stimulus vertical eccentricity (deg; Down(-) to Up(+))"])

    H_internal = H_ecc_field
    V_internal = -V_ecc_field

    mon, geom = load_or_ask_monitor()
    win = visual.Window(
        size=(geom["res_x"], geom["res_y"]), fullscr=True, monitor=mon,
        units="deg", color=0.5, colorSpace="rgb", allowGUI=False
    )

    if angle_set == 0:
        response_text = "Use the UP and DOWN arrow keys.\n\nUP = motion tilted upward\nDOWN = motion tilted downward"
    else:
        response_text = "Use the LEFT and RIGHT arrow keys.\n\nLEFT = motion tilted to the left\nRIGHT = motion tilted to the right"

    readme = (
        "READ ME / SETUP (Tilt Global)\n\n"
        "This program implements a feature-based attention (FBA) motion training task\n"
        "inspired by Huxlin Lab motion training paradigms (tilt-global RDK).\n\n"
        "- Keep your eyes strictly on the central fixation dot.\n"
        "- A brief line (pre-cue) will point from fixation to the stimulus location.\n"
        "- Then a cloud of moving dots will appear at that location for 500 ms.\n"
        "- All dots move in the SAME direction on each trial (no direction dispersion).\n"
        "- Your task is to report whether the GLOBAL motion is tilted up vs down (horizontal axis)\n"
        "  or left vs right (vertical axis).\n\n"
        f"Stimulus location in visual field (deg): H = {H_ecc_field:.1f}, V = {V_ecc_field:.1f}\n"
        "  (negative V = lower field, positive V = upper field)\n\n"
        "Training uses 3 interleaved staircases on tilt angle (deviation from the main axis).\n"
        "Each session includes the number of trials you select at startup (default: 300).\n\n"
        + response_text +
        "\n\nPress SPACE to start."
    )

    instr = visual.TextStim(win, text=readme, color=-1, height=0.6, wrapWidth=20)
    instr.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        win.close()
        return None

    summary_fba = run_fba_rdk_tilt_global(win, geom, subject_id, (H_internal, V_internal),
                                              angle_set=angle_set, total_trials=total_trials)

    if summary_fba is None:
        msg = (
            "Session ended early.\n\n"
            "No summary could be generated, but any collected data should be in the data folder.\n\n"
            "Press any key to exit."
        )
    else:
        if summary_fba.get("error"):
            msg = (
                "Session ended due to an unexpected error.\n\n"
                f"Trials saved: {summary_fba.get('n_trials', 0)}\n"
                "Please check the data folder for the CSV/JSON files.\n\n"
                "Press any key to exit."
            )
        elif summary_fba.get("aborted"):
            msg = (
                "Session interrupted.\n\n"
                f"Trials saved: {summary_fba.get('n_trials', 0)}\n"
                f"Accuracy so far: {summary_fba.get('accuracy_percent', float('nan')):.1f}%\n"
                f"Current threshold estimate: {summary_fba.get('final_threshold_deg', float('nan')):.2f} deg\n\n"
                "Press any key to exit."
            )
        else:
            msg = (
                "Training complete (Tilt Global)!\n\n"
                f"Accuracy: {summary_fba['accuracy_percent']:.1f}%\n"
                f"Final direction threshold (avg of 3 staircases): {summary_fba['final_threshold_deg']:.2f} deg\n\n"
                "Press any key to exit."
            )
    text = visual.TextStim(win, text=msg, color=-1, height=0.8)
    text.draw()
    win.flip()
    event.waitKeys()
    win.close()
    return None

if __name__ == "__main__":
    main()