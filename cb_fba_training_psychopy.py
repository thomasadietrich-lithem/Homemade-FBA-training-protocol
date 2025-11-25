# cb_fba_FBAonly_v4.py
# PsychoPy implementation of Huxlin Lab FBA motion training (direction-range RDK)
# - FBA task only (no Gabor)
# - Pre-cue line from fixation to stimulus location
# - Circular dots (like Psychtoolbox Screen('DrawDots', ..., 2))
# - Start beep + correct / incorrect beeps
# - Intro "READ ME" screen in English
#
# Coordinate convention for the user:
#   H_ecc_stim (deg): negative = left,   positive = right
#   V_ecc_stim (deg): negative = down,   positive = up   (visual field)

from psychopy import visual, core, event, gui, monitors, sound
import numpy as np
import os, csv, math, json, datetime

# ---------- Monitor / geometry helpers ----------

def get_screen_pixels():
    """Return (width_px, height_px)."""
    try:
        import pyglet
        d = pyglet.canvas.get_display()
        s = d.get_default_screen()
        return int(s.width), int(s.height)
    except Exception:
        # Fallback
        return 1920, 1080

def load_or_ask_monitor():
    """
    Ask for screen width (cm) and viewing distance (cm), cache in monitor_profiles.json
    per (system, resolution) profile.
    Returns (monitor, geom_info) where geom_info contains scale_factor etc.
    """
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
        # Ask user once per screen setup
        dlg_dict = {
            "Screen width (cm)": 61.0,
            "Viewing distance (cm)": 42.0,
        }
        dlg = gui.DlgFromDict(dlg_dict, title="Screen calibration",
                              order=["Screen width (cm)", "Viewing distance (cm)"])
        if not dlg.OK:
            core.quit()
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

    # Build PsychoPy monitor
    mon = monitors.Monitor(ident)
    mon.setWidth(prof["width_cm"])
    mon.setDistance(prof["distance_cm"])
    mon.setSizePix(prof["size_pix"])

    # Geometry compatible with Matlab scale_factor
    screen_width_cm = prof["width_cm"]
    viewing_dist_cm = prof["distance_cm"]
    res_x = prof["size_pix"][0]

    theta_deg = math.degrees(math.atan((screen_width_cm/2.0)/viewing_dist_cm))
    # arcmin per pixel (Matlab scale_factor)
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
    """Convert visual angle in degrees to pixels using Matlab-style scale_factor."""
    arcmin = deg * 60.0
    return arcmin / geom["arcmin_per_pix"]

# ---------- Core FBA RDK training (Rochester-like) ----------

def run_fba_rdk(win, geom, subject_id, location_deg_internal, angle_set=0,
                n_staircases=3, n_trials_per_staircase=100,
                save_dir="data"):
    """
    Reimplementation of the Rochester FBA RDK training.

    angle_set: 0 = horizontal axis (respond UP/DOWN),
               1 = vertical axis   (respond LEFT/RIGHT)

    location_deg_internal: (H_ecc_stim, V_ecc_stim_internal),
      internal convention: negative V = up, positive V = down (screen coords),
      already converted from visual-field convention before calling.
    """

    # --- Parameters from TrainingSetup.m ---
    H_ecc_fix = 0.0
    V_ecc_fix = 0.0
    H_ecc_stim, V_ecc_stim_internal = location_deg_internal

    cue_duration = 0.2       # sec
    cue_color = [1, 1, 1]

    stimulus_duration_ms = 500.0
    aperture_radius_deg = 2.5
    dot_density = 3.5        # dots/deg^2
    initial_dot_size_arcmin = 14.0
    dot_color = -1.0         # grayscale (background = 0.5)
    dot_speed_deg_per_s = 10.0
    dot_lifetime_ms = 200.0

    background = 0.5         # mid-grey

    angle_range = [85, 53.1, 33.2, 20.75, 12.97, 8.1, 5.1, 3.2, 2.0, 1.2, 0.8, 0.5]
    stair1 = 1
    stair2 = 4
    stair3 = 8

    total_trials = n_staircases * n_trials_per_staircase

    # --- Refresh rate ---
    refresh = win.getActualFrameRate(nIdentical=20, nMaxFrames=120, nWarmUpFrames=20)
    if refresh is None or refresh <= 0:
        refresh = 60.0
    mv_length = int(round(stimulus_duration_ms / (1000.0/refresh)))
    lifetime_frames = int(round(dot_lifetime_ms / (1000.0/refresh)))

    # Positions in degrees
    fix_x_deg = H_ecc_fix
    fix_y_deg = V_ecc_fix
    stim_x_deg = H_ecc_stim

    # Dot size in pixels (arcmin / scale_factor)
    dot_size_pix = int(math.floor(initial_dot_size_arcmin / geom["arcmin_per_pix"]))
    if dot_size_pix < 2:
        dot_size_pix = 2

    # Number of dots according to density
    area_deg2 = math.pi * (aperture_radius_deg ** 2)
    n_dots = int(round(dot_density * area_deg2))

    # Step in deg/frame
    dot_step_deg = dot_speed_deg_per_s / refresh

    # Fixation
    fixation = visual.Circle(win, radius=0.1, fillColor=-1, lineColor=-1,
                             pos=(fix_x_deg, fix_y_deg), units="deg")
    fixation_inner = visual.Circle(win, radius=0.05, fillColor=1, lineColor=1,
                                   pos=(fix_x_deg, fix_y_deg), units="deg")

    # RDK centre in pixels (internal convention already applied to V)
    stim_x_pix = deg_to_pix(stim_x_deg, geom)
    stim_y_pix = -deg_to_pix(V_ecc_stim_internal, geom)  # positive internal -> down

    # Dots as ElementArrayStim with circular mask
    dots = visual.ElementArrayStim(
        win,
        nElements=n_dots,
        elementTex=None,
        elementMask="circle",          # <-- CIRCULAR DOTS
        xys=np.zeros((n_dots, 2)),
        sizes=[dot_size_pix] * n_dots,
        units="pix",                   # positions are in pixels
        colors=[dot_color] * n_dots,
        colorSpace="rgb",
        sfs=0
    )

    # Staircase bookkeeping
    stair_array = []
    for s in range(n_staircases):
        stair_array.extend([s+1]*n_trials_per_staircase)
    np.random.shuffle(stair_array)

    staircount1 = staircount2 = staircount3 = 0

    results = []  # trial data
    trial = 0
    clock = core.Clock()

    # Sound feedback (start + correct / incorrect)
    snd_start = sound.Sound(value=1000, secs=0.05)
    snd_correct = sound.Sound(value=1200, secs=0.12)   # high beep
    snd_incorrect = sound.Sound(value=800, secs=0.12)  # low beep

    # Background
    win.color = background
    win.flip()

    while trial < total_trials:
        trial += 1
        which_stair = stair_array[trial-1]
        if which_stair == 1:
            angle_deviationP = angle_range[stair1-1]
        elif which_stair == 2:
            angle_deviationP = angle_range[stair2-1]
        else:
            angle_deviationP = angle_range[stair3-1]

        # direction (1 or 2) + orientation (-1 or +1)
        direction = np.random.randint(1, 3)
        orientation = np.random.choice([-1, 1])

        # Determine correct/incorrect keys and base angle exactly like Matlab
        if direction == 2:
            if angle_set == 0:
                # horizontal axis, respond up/down
                if orientation == 1:
                    correct_key = "up"
                    incorrect_key = "down"
                else:
                    correct_key = "down"
                    incorrect_key = "up"
            else:
                # vertical axis, respond left/right
                if orientation == -1:
                    correct_key = "left"
                    incorrect_key = "right"
                else:
                    correct_key = "right"
                    incorrect_key = "left"
        else:  # direction == 1
            if angle_set == 0:
                if orientation == 1:
                    incorrect_key = "up"
                    correct_key = "down"
                else:
                    incorrect_key = "down"
                    correct_key = "up"
            else:
                if orientation == -1:
                    incorrect_key = "left"
                    correct_key = "right"
                else:
                    incorrect_key = "right"
                    correct_key = "left"

        # Motion angle in degrees
        if angle_set == 0 and direction == 1:
            angle_deg = 0 + angle_deviationP * orientation
        elif angle_set == 0 and direction == 2:
            angle_deg = 180 + angle_deviationP * orientation
        elif angle_set == 1 and direction == 1:
            angle_deg = 270 + angle_deviationP * orientation
        else:  # angle_set == 1 and direction == 2
            angle_deg = 90 + angle_deviationP * orientation

        # ---------- Pre-cue: line fixation -> stimulus ----------
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

        # Remove cue, fixation only
        fixation.draw()
        fixation_inner.draw()
        win.flip()
        core.wait(0.05)

        # ---------- Initialize dots ----------
        stimulus_radius_pix = deg_to_pix(aperture_radius_deg, geom)
        positions = np.zeros((mv_length, n_dots, 2), dtype=float)
        ages = np.zeros((mv_length, n_dots), dtype=int)

        # First frame: random positions in circular aperture
        for i in range(n_dots):
            while True:
                x = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                y = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                if x*x + y*y <= stimulus_radius_pix**2:
                    positions[0, i, 0] = x
                    positions[0, i, 1] = y
                    ages[0, i] = np.random.randint(1, lifetime_frames+1)
                    break

        # Velocity in pixels/frame
        step_pix = deg_to_pix(dot_step_deg, geom)
        angle_rad = math.radians(angle_deg)
        vx = step_pix * math.cos(angle_rad)
        vy = step_pix * math.sin(angle_rad)

        # Frame-by-frame update
        for f in range(1, mv_length):
            for i in range(n_dots):
                x = positions[f-1, i, 0] + vx
                y = positions[f-1, i, 1] + vy
                age = ages[f-1, i] + 1

                # Lifetime reset
                if age > lifetime_frames:
                    while True:
                        rx = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                        ry = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                        if rx*rx + ry*ry <= stimulus_radius_pix**2:
                            x, y = rx, ry
                            age = 1
                            break

                # Wrap around aperture (torus)
                if x > stimulus_radius_pix:
                    x -= 2*stimulus_radius_pix
                elif x < -stimulus_radius_pix:
                    x += 2*stimulus_radius_pix
                if y > stimulus_radius_pix:
                    y -= 2*stimulus_radius_pix
                elif y < -stimulus_radius_pix:
                    y += 2*stimulus_radius_pix

                # Re-draw if outside circle
                if x*x + y*y > stimulus_radius_pix**2:
                    while True:
                        rx = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                        ry = (np.random.rand() - 0.5) * 2 * stimulus_radius_pix
                        if rx*rx + ry*ry <= stimulus_radius_pix**2:
                            x, y = rx, ry
                            break

                positions[f, i, 0] = x
                positions[f, i, 1] = y
                ages[f, i] = age

        # ---------- Start beep ----------
        snd_start.play()
        core.wait(0.05)

        # ---------- Play RDK ----------
        win.callOnFlip(event.clearEvents, eventType="keyboard")
        win.flip()
        clock.reset()
        for f in range(mv_length):
            xys = positions[f].copy()
            xys[:, 0] += stim_x_pix
            xys[:, 1] += stim_y_pix
            dots.xys = xys
            dots.draw()
            fixation.draw()
            fixation_inner.draw()
            win.flip()

        # Post-stim fixation
        fixation.draw()
        fixation_inner.draw()
        win.flip()

        # ---------- Response ----------
        rt = None
        correct = 0
        keys = event.waitKeys(maxWait=2.0,
                              keyList=[correct_key, incorrect_key, "escape"],
                              timeStamped=clock)
        if keys:
            key, rt = keys[0]
            if key == "escape":
                win.close()
                core.quit()
            if key == correct_key:
                correct = 1
                snd_correct.play()
            else:
                correct = 0
                snd_incorrect.play()

        # ---------- Staircase ----------
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

        # ITI fixation
        fixation.draw()
        fixation_inner.draw()
        win.flip()
        core.wait(0.5)

    # ---------- Analytics ----------
    correct_trials = sum(r[6] for r in results)
    accuracy = 100.0 * correct_trials / len(results)
    final_thresh = (angle_range[stair1-1] +
                    angle_range[stair2-1] +
                    angle_range[stair3-1]) / 3.0

    # Save
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{subject_id}_FBA_{ts}"
    csv_path = os.path.join(save_dir, base + "_trials.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "angle_dev_deg", "staircase", "direction_code",
                    "orientation_sign", "rt_s", "correct", "angle_deg"])
        for row in results:
            w.writerow(row)

    summary_path = os.path.join(save_dir, base + "_summary.json")
    summary = {
        "subject": subject_id,
        "task": "FBA_RDK",
        "location_deg_internal": {"H": H_ecc_stim, "V_internal": V_ecc_stim_internal},
        "angle_set": angle_set,
        "n_trials": len(results),
        "accuracy_percent": accuracy,
        "final_threshold_deg": final_thresh,
        "stair_levels": {"stair1": stair1, "stair2": stair2, "stair3": stair3},
        "angle_range": angle_range,
        "timestamp": ts,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

# ---------- Main entry point ----------

def main():
    # Main dialog
    info = {
        "Subject ID": "TEST01",
        "Angle set (0=horizontal axis/UP-DOWN, 1=vertical axis/LEFT-RIGHT)": 0,
        "H_ecc_stim (deg, - left + right)": -3.0,
        "V_ecc_stim (deg, - down + up)": -5.0,
    }
    dlg = gui.DlgFromDict(info, title="CB training (Rochester-style, FBA only)")
    if not dlg.OK:
        core.quit()

    subject_id = str(info["Subject ID"])
    angle_set = int(info["Angle set (0=horizontal axis/UP-DOWN, 1=vertical axis/LEFT-RIGHT)"])
    H_ecc_field = float(info["H_ecc_stim (deg, - left + right)"])
    V_ecc_field = float(info["V_ecc_stim (deg, - down + up)"])

    # User convention (visual field):
    #   V < 0 = lower field (down), V > 0 = upper field (up)
    # Internal convention (screen coords) needs the opposite sign in deg_to_pix
    H_internal = H_ecc_field
    V_internal = -V_ecc_field

    mon, geom = load_or_ask_monitor()
    win = visual.Window(
        size=(geom["res_x"], geom["res_y"]), fullscr=True, monitor=mon,
        units="deg", color=0.5, colorSpace="rgb", allowGUI=False
    )

    # ---------- READ ME / instructions screen ----------
    if angle_set == 0:
        response_text = "Use the UP and DOWN arrow keys.\n\nUP = motion tilted upward\nDOWN = motion tilted downward"
    else:
        response_text = "Use the LEFT and RIGHT arrow keys.\n\nLEFT = motion tilted to the left\nRIGHT = motion tilted to the right"

    readme = (
        "READ ME / SETUP\n\n"
        "This program implements a feature-based attention (FBA) motion training task\n"
        "based on the Huxlin Lab protocol (direction-range RDK).\n\n"
        "- Keep your eyes strictly on the central fixation dot.\n"
        "- A brief line (pre-cue) will point from fixation to the stimulus location.\n"
        "- Then a cloud of moving dots will appear at that location for 500 ms.\n"
        "- Your task is to report the direction of motion relative to the main axis.\n\n"
        f"Stimulus location in visual field (deg): H = {H_ecc_field:.1f}, V = {V_ecc_field:.1f}\n"
        "  (negative V = lower field, positive V = upper field)\n\n"
        "Training uses 3 interleaved staircases on direction range (angle deviation).\n"
        "Each session includes 300 trials (about 10–15 minutes).\n\n"
        + response_text +
        "\n\nPress SPACE to start."
    )

    instr = visual.TextStim(win, text=readme, color=-1, height=0.6, wrapWidth=20)
    instr.draw()
    win.flip()
    keys = event.waitKeys(keyList=["space", "escape"])
    if "escape" in keys:
        win.close()
        core.quit()

    # ---------- Run FBA training ----------
    summary_fba = run_fba_rdk(win, geom, subject_id, (H_internal, V_internal),
                              angle_set=angle_set)

    # ---------- End screen ----------
    msg = (
        "Training complete!\n\n"
        f"Accuracy: {summary_fba['accuracy_percent']:.1f}%\n"
        f"Final direction threshold (avg of 3 staircases): {summary_fba['final_threshold_deg']:.2f} deg\n\n"
        "Press any key to exit."
    )
    text = visual.TextStim(win, text=msg, color=-1, height=0.8)
    text.draw()
    win.flip()
    event.waitKeys()
    win.close()
    core.quit()

if __name__ == "__main__":
    main()
