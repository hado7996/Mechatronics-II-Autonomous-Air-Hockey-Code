#Changes made: fitted straight wall lines for red/orange zone indicators
# 4/22/26 changes, intercept history length from 1-4
# paddle y history 56->8
# P2 skipped when puck moving away from goal

import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
from collections import deque
import serial
import serial.tools.list_ports

# -------------------------
# RealSense setup
# -------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
pipeline.start(config)

# -------------------------
# Arduino Serial Setup
# -------------------------
SERIAL_PORT = None
ports = serial.tools.list_ports.comports()
for port in ports:
    if "ttyACM" in port.device or "ttyUSB" in port.device:
        SERIAL_PORT = port.device
        break
if SERIAL_PORT is None:
    raise Exception("Arduino not found")
ser = serial.Serial(SERIAL_PORT, 115200)
time.sleep(2)
print(f"Connected to Arduino on {SERIAL_PORT}")

# -------------------------
# PID gains — tune these
# -------------------------
KP    = 0.18
KI    = 0.0
KD    = 0.024   #0.024
I_MAX = 0.0

# -------------------------
# Output limits
# -------------------------
MAX_PWM_OUT = 95 #25
DEADBAND    = 2 #5

# -------------------------
# X-axis boundary enforcement
# -------------------------
HARD_STOP_PX = 10
SLOWDOWN_PX  = 30

# -------------------------
# Enemy proximity threshold
# -------------------------
ENEMY_PUCK_CLOSE_PX = 80

# -------------------------
# PID state
# -------------------------
integrator    = 0.0
prev_error    = 0
prev_pid_time = time.time()

def compute_pid(error, dt):
    global integrator, prev_error

    if abs(error) <= DEADBAND:
        integrator = 0.0
        prev_error = 0
        return 0

    integrator += error * dt
    integrator  = max(-I_MAX, min(I_MAX, integrator))

    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    prev_error  = error

    output = KP * error + KI * integrator + KD * derivative
    output = max(-MAX_PWM_OUT, min(MAX_PWM_OUT, output))

    return int(output)

def apply_boundary_scaling(pwm, paddle_y, min_y, max_y):
    if paddle_y is None or (min_y == 0 and max_y == 0):
        return pwm

    dist = (paddle_y - min_y) if pwm < 0 else (max_y - paddle_y)

    if dist <= HARD_STOP_PX:
        return 0
    if dist >= SLOWDOWN_PX:
        return pwm

    factor = (dist - HARD_STOP_PX) / (SLOWDOWN_PX - HARD_STOP_PX)
    return int(pwm * factor)

# -------------------------
# Tracking state
# -------------------------
last_sent_pwm            = 0
TARGET_REACHED_THRESHOLD = 5

VELOCITY_ROLLING_FRAMES = 2
puck_history          = deque(maxlen=VELOCITY_ROLLING_FRAMES)
time_history          = deque(maxlen=VELOCITY_ROLLING_FRAMES)
continuous_detections = 0
MIN_FRAMES_FOR_VELOCITY = VELOCITY_ROLLING_FRAMES
VELOCITY_THRESHOLD    = 35 #35

velocity_history  = deque(maxlen=VELOCITY_ROLLING_FRAMES)
paddle_y_history  = deque(maxlen=5) #5->8
intercept_history = deque(maxlen=1) #1->4

# -------------------------
# Precomputed HSV ranges and kernels
# -------------------------
lower_yellow = np.array([15, 90, 70])
upper_yellow = np.array([36, 255, 210])
lower_red1   = np.array([0, 140, 80])
upper_red1   = np.array([10, 255, 255])
lower_red2   = np.array([165, 140, 80])
upper_red2   = np.array([180, 255, 255])
lower_green  = np.array([35, 120, 30])
upper_green  = np.array([90, 255, 105])
lower_blue   = np.array([100, 80, 50])
upper_blue   = np.array([130, 255, 255])
lower_purple = np.array([115, 90, 30])
upper_purple = np.array([140, 190, 137])

morph_kernel  = np.ones((3, 3), np.uint8)
yellow_kernel = np.ones((4, 4), np.uint8)

# -------------------------
# Timestamp helper
# -------------------------
def ts():
    return time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"

# -------------------------
# Boundary helpers
# -------------------------
def densify_contour(contour, max_gap=4):
    dense = []
    pts = contour[:, 0, :]
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        dense.append(p1)
        dist = np.linalg.norm(p2.astype(float) - p1.astype(float))
        if dist > max_gap:
            steps = int(dist / max_gap)
            for s in range(1, steps):
                t = s / steps
                interp = (p1 + t * (p2.astype(float) - p1.astype(float))).astype(int)
                dense.append(interp)
    return np.array(dense, dtype=np.int32).reshape(-1, 1, 2)

def contour_centroid(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

# -------------------------
# Fit straight line y = mx + b through wall points
# -------------------------
def fit_wall_line(pts):
    if len(pts) < 2:
        return (0.0, float(pts[0][1]) if len(pts) == 1 else 0.0)
    xs = pts[:, 0].astype(float)
    ys = pts[:, 1].astype(float)
    coeffs = np.polyfit(xs, ys, 1)
    return (float(coeffs[0]), float(coeffs[1]))

def wall_y_at_x(m, b, x):
    return int(round(m * x + b))

# -------------------------
# Trajectory prediction
# -------------------------
def predict_trajectory(x, y, vx, vy, min_x, max_x, min_y, max_y):
    points = []

    if vx == 0 and vy == 0:
        return points

    MIN_SPEED = 50.0
    speed = math.sqrt(vx**2 + vy**2)
    if speed < MIN_SPEED:
        return points

    dx = vx / speed
    dy = vy / speed

    TRAVEL_DIST    = (max_x - min_x) * 4.0
    remaining_dist = TRAVEL_DIST

    curr_x = max(min_x, min(max_x, float(x)))
    curr_y = max(min_y, min(max_y, float(y)))

    points.append((int(curr_x), int(curr_y)))

    t_vals = []
    if dx > 0:
        t_vals.append((max_x - curr_x) / dx)
    elif dx < 0:
        t_vals.append((min_x - curr_x) / dx)
    if dy > 0:
        t_vals.append((max_y - curr_y) / dy)
    elif dy < 0:
        t_vals.append((min_y - curr_y) / dy)

    t_vals = [t for t in t_vals if t > 0]
    if not t_vals:
        return points

    t_first      = min(t_vals)
    dist_to_wall = t_first

    if remaining_dist <= dist_to_wall:
        x_end = curr_x + dx * remaining_dist
        y_end = curr_y + dy * remaining_dist
        x_end = max(min_x, min(max_x, x_end))
        y_end = max(min_y, min(max_y, y_end))
        points.append((int(x_end), int(y_end)))
        return points

    x1 = curr_x + dx * dist_to_wall
    y1 = curr_y + dy * dist_to_wall
    x1 = max(min_x, min(max_x, x1))
    y1 = max(min_y, min(max_y, y1))
    points.append((int(x1), int(y1)))

    remaining_dist -= dist_to_wall

    hit_vertical   = abs(x1 - min_x) < 2 or abs(x1 - max_x) < 2
    hit_horizontal = abs(y1 - min_y) < 2 or abs(y1 - max_y) < 2
    if hit_vertical:
        dx = -dx
    if hit_horizontal:
        dy = -dy

    x2 = x1 + dx * remaining_dist
    y2 = y1 + dy * remaining_dist
    x2 = max(min_x, min(max_x, x2))
    y2 = max(min_y, min(max_y, y2))
    points.append((int(x2), int(y2)))

    return points


def trajectory_intercept_at_x(traj_pts, target_x, frame_h):
    for i in range(len(traj_pts) - 1):
        x0, y0 = traj_pts[i]
        x1, y1 = traj_pts[i + 1]
        seg_dx = x1 - x0
        if seg_dx == 0:
            continue
        t = (target_x - x0) / seg_dx
        if 0 <= t <= 1:
            intercept_y = int(y0 + t * (y1 - y0))
            if 0 <= intercept_y < frame_h:
                return intercept_y
    return None


# -------------------------
# Boundary + goal state
# -------------------------
BOUNDARY_FREEZE_SECONDS = 5
frozen_poly = None
frozen_goal = None
min_x = min_y = max_x = max_y = 0

# Fitted wall line params (set at freeze time)
# top wall:    y = top_m * x + top_b
# bottom wall: y = bottom_m * x + bottom_b
top_m    = top_b    = 0.0
bottom_m = bottom_b = 0.0
wall_x1  = wall_x2  = 0

start_time = time.time()

print("Starting... boundary will freeze after 5 seconds.")

cv2.namedWindow("Detection")

try:
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame        = np.asanyarray(color_frame.get_data())
        current_time = time.time()
        elapsed      = current_time - start_time

        dt            = current_time - prev_pid_time
        dt            = max(0.001, min(dt, 0.1))
        prev_pid_time = current_time

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------------------------
        # Yellow boundary detection
        # -------------------------
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  yellow_kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, yellow_kernel)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 400]

        if frozen_poly is None:
            if large_contours:
                all_pts = np.vstack(large_contours)
                cx_all  = int(np.mean(all_pts[:, 0, 0]))
                cy_all  = int(np.mean(all_pts[:, 0, 1]))

                hull       = cv2.convexHull(all_pts)
                hull_dense = densify_contour(hull, max_gap=4)

                SHRINK_PX  = 14
                inner_hull = []
                for pt in hull_dense:
                    px, py = pt[0]
                    dx     = px - cx_all
                    dy     = py - cy_all
                    dist   = np.sqrt(dx**2 + dy**2)
                    if dist > 0:
                        px_new = px - int(SHRINK_PX * dx / dist)
                        py_new = py - int(SHRINK_PX * dy / dist)
                    else:
                        px_new, py_new = px, py
                    inner_hull.append([[px_new, py_new]])

                inner_hull = np.array(inner_hull, dtype=np.int32)
                inner_poly = cv2.approxPolyDP(inner_hull, 2, closed=True)

                cv2.polylines(frame, [inner_poly], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.drawContours(frame, large_contours, -1, (0, 200, 255), 1)

                if elapsed >= BOUNDARY_FREEZE_SECONDS:
                    frozen_poly = inner_poly
                    bx, by, bw, bh = cv2.boundingRect(frozen_poly)
                    min_x, min_y, max_x, max_y = bx, by, bx + bw, by + bh
                    print(f"[{ts()}] Boundary frozen! min_x={min_x} max_x={max_x}")

                    poly_pts  = frozen_poly[:, 0, :]
                    quarter_h = (max_y - min_y) * 0.25

                    top_pts    = poly_pts[poly_pts[:, 1] < min_y + quarter_h]
                    bottom_pts = poly_pts[poly_pts[:, 1] > max_y - quarter_h]

                    if len(top_pts) >= 2:
                        top_m, top_b = fit_wall_line(top_pts)
                    else:
                        top_m, top_b = 0.0, float(min_y)

                    if len(bottom_pts) >= 2:
                        bottom_m, bottom_b = fit_wall_line(bottom_pts)
                    else:
                        bottom_m, bottom_b = 0.0, float(max_y)

                    wall_x1 = min_x
                    wall_x2 = max_x

                    print(f"[{ts()}] Top wall:    y = {top_m:.4f}x + {top_b:.1f}")
                    print(f"[{ts()}] Bottom wall: y = {bottom_m:.4f}x + {bottom_b:.1f}")

            seconds_left = max(0, BOUNDARY_FREEZE_SECONDS - elapsed)
            cv2.putText(frame, f"Boundary freezing in {seconds_left:.1f}s",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # -------------------------
        # Red puck detection
        # -------------------------
        mask1    = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2    = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN,  morph_kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel)

        puck_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        puck_x = puck_y = None
        paddle_x = paddle_y = None

        for cnt in puck_contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    puck_x = int(M["m10"] / M["m00"])
                    puck_y = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
                    cv2.circle(frame, (puck_x, puck_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "PUCK", (puck_x - 30, puck_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                break

        # -------------------------
        # Green paddle detection
        # -------------------------
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  morph_kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, morph_kernel)

        paddle_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in paddle_contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    paddle_x     = int(M["m10"] / M["m00"])
                    paddle_y_raw = int(M["m01"] / M["m00"])
                    paddle_y_history.append(paddle_y_raw)
                    paddle_y = int(np.mean(paddle_y_history))
                    cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (paddle_x, paddle_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, "PADDLE", (paddle_x - 30, paddle_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

        if paddle_x is not None:
            cv2.line(frame, (paddle_x, 0), (paddle_x, frame.shape[0]), (255, 255, 0), 2)

        # -------------------------
        # Purple enemy paddle detection
        # -------------------------
        purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_OPEN,  morph_kernel)
        purple_mask = cv2.morphologyEx(purple_mask, cv2.MORPH_CLOSE, morph_kernel)

        purple_contours, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        enemy_x = enemy_y = None
        for cnt in purple_contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    enemy_x = int(M["m10"] / M["m00"])
                    enemy_y = int(M["m01"] / M["m00"])
                    cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)
                    cv2.circle(frame, (enemy_x, enemy_y), 5, (255, 0, 255), -1)
                    cv2.putText(frame, "ENEMY", (enemy_x - 30, enemy_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                break

        # -------------------------
        # Blue goal detection — largest contour, freezes once found
        # -------------------------
        if frozen_goal is None:
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN,  morph_kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morph_kernel)

            cv2.imshow("Blue Mask", blue_mask)

            blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blue_contours = [c for c in blue_contours if cv2.contourArea(c) > 300]

            if blue_contours:
                best        = max(blue_contours, key=cv2.contourArea)
                frozen_goal = contour_centroid(best)
                if frozen_goal is not None:
                    print(f"[{ts()}] Goal frozen at {frozen_goal}")

        # Draw frozen goal and puck->goal orange alignment line
        if frozen_goal is not None:
            cv2.circle(frame, frozen_goal, 8, (255, 0, 0), -1)
            cv2.putText(frame, "GOAL", (frozen_goal[0] - 20, frozen_goal[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if puck_x is not None:
                cv2.line(frame, (puck_x, puck_y), frozen_goal, (0, 165, 255), 2)

        # -------------------------
        # Puck history
        # -------------------------
        if puck_x is not None:
            puck_history.append((puck_x, puck_y))
            time_history.append(current_time)
            continuous_detections += 1
        else:
            continuous_detections = 0
            puck_history.clear()
            time_history.clear()

        puck_history = deque(puck_history, maxlen=10)
        time_history = deque(time_history, maxlen=10)
        if continuous_detections >= 10:
            continuous_detections = 10

        # -------------------------
        # Velocity calculation
        # -------------------------
        vx = vy = 0.0

        if continuous_detections >= 2:
            n            = len(puck_history)
            samples      = 10
            xi,      yi      = puck_history[n - 1]
            xi_prev, yi_prev = puck_history[n - 2]
            ti               = time_history[n - 1]
            ti_prev          = time_history[n - 2]
            frame_dt         = ti - ti_prev

            if frame_dt > 0:
                inst_vx = (xi - xi_prev) / frame_dt
                inst_vy = (yi - yi_prev) / frame_dt
                velocity_history.append((inst_vx, inst_vy))
                velocity_history = deque(velocity_history, maxlen=samples)

                if len(velocity_history) >= MIN_FRAMES_FOR_VELOCITY:
                    vx = float(np.mean([v[0] for v in velocity_history]))
                    vy = float(np.mean([v[1] for v in velocity_history]))
                else:
                    vx, vy = inst_vx, inst_vy

                speed = math.sqrt(vx**2 + vy**2)

                if speed > VELOCITY_THRESHOLD:
                    scale = 0.05
                    cv2.arrowedLine(frame, (puck_x, puck_y),
                                    (int(puck_x + vx * scale), int(puck_y + vy * scale)),
                                    (0, 255, 0), 3)

                cv2.putText(frame, f"Velocity: ({int(vx)}, {int(vy)}) px/s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------------------------
        # Compute puck-to-enemy distance
        # -------------------------
        puck_enemy_dist = None
        if puck_x is not None and enemy_x is not None:
            puck_enemy_dist = math.sqrt((puck_x - enemy_x)**2 + (puck_y - enemy_y)**2)

        # -------------------------
        # Determine if puck is moving away from goal
        # True when vx dot (goal - puck) is negative — puck moving opposite to goal direction
        # -------------------------
        puck_moving_away_from_goal = False
        if frozen_goal is not None and puck_x is not None and abs(vx) > 0:
            goal_dx = frozen_goal[0] - puck_x
            puck_moving_away_from_goal = (vx * goal_dx) < 0

        # -------------------------
        # Compute pixel error → PID → boundary scale → send
        # -------------------------
        target_y  = None
        raw_error = None

        if puck_y is not None and paddle_y is not None:

            # ----------------------------------------------------------
            # Priority 1: Enemy close to puck — bounced enemy→puck path
            # ----------------------------------------------------------
            if (enemy_x is not None and puck_x is not None and
                    puck_enemy_dist is not None and
                    puck_enemy_dist < ENEMY_PUCK_CLOSE_PX and
                    frozen_poly is not None and min_x != max_x and min_y != max_y):

                edx = puck_x - enemy_x
                edy = puck_y - enemy_y
                e_speed = math.sqrt(edx**2 + edy**2)

                if e_speed > 0:
                    VIRTUAL_SPEED = 400.0
                    evx = (edx / e_speed) * VIRTUAL_SPEED
                    evy = (edy / e_speed) * VIRTUAL_SPEED

                    traj_pts = predict_trajectory(
                        puck_x, puck_y, evx, evy,
                        min_x, max_x, min_y, max_y)

                    for i in range(len(traj_pts) - 1):
                        cv2.line(frame, traj_pts[i], traj_pts[i + 1], (255, 0, 255), 2)

                    if paddle_x is not None and len(traj_pts) >= 2:
                        intercept_y = trajectory_intercept_at_x(traj_pts, paddle_x, frame.shape[0])
                        if intercept_y is not None:
                            cv2.circle(frame, (paddle_x, intercept_y), 10, (255, 0, 255), -1)
                            cv2.putText(frame, "INTERCEPT", (paddle_x + 12, intercept_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            intercept_history.append(intercept_y)
                            target_y  = int(np.mean(intercept_history))
                            raw_error = target_y - paddle_y
                        else:
                            intercept_y = traj_pts[-1][1]
                            raw_error   = intercept_y - paddle_y
                        print(f"[{ts()}] MODE: P1 enemy bounce | dist={puck_enemy_dist:.0f} intercept_y={intercept_y} paddle_y={paddle_y} raw_err={raw_error:+d}")

            # ----------------------------------------------------------
            # Priority 2: Puck far from enemy AND moving toward goal
            # Skipped if puck is moving away from goal — falls to P3
            # ----------------------------------------------------------
            if raw_error is None:
                speed = math.sqrt(vx**2 + vy**2)
                if (speed > VELOCITY_THRESHOLD and
                        not puck_moving_away_from_goal and
                        frozen_poly is not None and min_x != max_x and min_y != max_y and
                        puck_x is not None):

                    traj_pts = predict_trajectory(
                        puck_x, puck_y, vx, vy,
                        min_x, max_x, min_y, max_y)

                    for i in range(len(traj_pts) - 1):
                        cv2.line(frame, traj_pts[i], traj_pts[i + 1], (255, 0, 255), 2)

                    if paddle_x is not None and len(traj_pts) >= 2:
                        intercept_y = trajectory_intercept_at_x(traj_pts, paddle_x, frame.shape[0])
                        if intercept_y is not None:
                            cv2.circle(frame, (paddle_x, intercept_y), 10, (255, 0, 255), -1)
                            intercept_history.append(intercept_y)
                            target_y  = int(np.mean(intercept_history))
                            raw_error = target_y - paddle_y
                            print(f"[{ts()}] MODE: P2 velocity bounce | vx={int(vx)} vy={int(vy)} intercept_y={intercept_y} paddle_y={paddle_y} raw_err={raw_error:+d}")
                        else:
                            intercept_y = traj_pts[-1][1]
                            raw_error   = intercept_y - paddle_y
                            print(f"[{ts()}] MODE: P2 velocity bounce (no cross) | intercept_y={intercept_y} paddle_y={paddle_y} raw_err={raw_error:+d}")

            # ----------------------------------------------------------
            # Priority 3: Fallback — block puck→goal straight line
            # Also used when puck is moving away from goal
            # ----------------------------------------------------------
            if raw_error is None:
                if puck_x is not None and frozen_goal is not None and paddle_x is not None:
                    dx = frozen_goal[0] - puck_x
                    dy = frozen_goal[1] - puck_y
                    if dx != 0:
                        t                = (paddle_x - puck_x) / dx
                        line_y_at_paddle = int(puck_y + t * dy)
                        raw_error        = line_y_at_paddle - paddle_y
                        print(f"[{ts()}] MODE: P3 block line | line_y={line_y_at_paddle} paddle_y={paddle_y} away={puck_moving_away_from_goal}")
                    else:
                        raw_error = puck_y - paddle_y
                        print(f"[{ts()}] MODE: P3 fallback | puck_y={puck_y} paddle_y={paddle_y}")
                else:
                    raw_error = puck_y - paddle_y
                    print(f"[{ts()}] MODE: P3 puck track | puck_y={puck_y} paddle_y={paddle_y}")

            # PID → signed PWM
            pwm = compute_pid(raw_error, dt)

            # X-axis boundary scaling
            if frozen_poly is not None:
                pwm = apply_boundary_scaling(pwm, paddle_y, min_y, max_y)

            # Draw boundary zone indicators — red hard stop, orange slowdown
            if frozen_poly is not None and paddle_x is not None and wall_x1 != wall_x2:
                cv2.line(frame,
                         (wall_x1, wall_y_at_x(top_m, top_b + HARD_STOP_PX, wall_x1)),
                         (wall_x2, wall_y_at_x(top_m, top_b + HARD_STOP_PX, wall_x2)),
                         (0, 0, 255), 1)
                cv2.line(frame,
                         (wall_x1, wall_y_at_x(top_m, top_b + SLOWDOWN_PX, wall_x1)),
                         (wall_x2, wall_y_at_x(top_m, top_b + SLOWDOWN_PX, wall_x2)),
                         (0, 165, 255), 1)
                cv2.line(frame,
                         (wall_x1, wall_y_at_x(bottom_m, bottom_b - HARD_STOP_PX, wall_x1)),
                         (wall_x2, wall_y_at_x(bottom_m, bottom_b - HARD_STOP_PX, wall_x2)),
                         (0, 0, 255), 1)
                cv2.line(frame,
                         (wall_x1, wall_y_at_x(bottom_m, bottom_b - SLOWDOWN_PX, wall_x1)),
                         (wall_x2, wall_y_at_x(bottom_m, bottom_b - SLOWDOWN_PX, wall_x2)),
                         (0, 165, 255), 1)

            if abs(raw_error) <= TARGET_REACHED_THRESHOLD:
                pwm        = 0
                integrator = 0.0

            ser.write((str(pwm) + "\n").encode())
            last_sent_pwm = pwm
            print(f"[{ts()}] raw_err={raw_error:+d} pwm={pwm:+d}")

        else:
            if last_sent_pwm != 0:
                ser.write(b"0\n")
                last_sent_pwm = 0
                integrator    = 0.0
                print(f"[{ts()}] No target | Sent: 0")

        # Draw frozen boundary — blue
        if frozen_poly is not None:
            cv2.polylines(frame, [frozen_poly], isClosed=True, color=(255, 0, 0), thickness=2)

        # Draw enemy proximity circle for debug
        if enemy_x is not None and puck_x is not None:
            cv2.circle(frame, (enemy_x, enemy_y), ENEMY_PUCK_CLOSE_PX, (255, 0, 255), 1)
            dist_label = f"d={puck_enemy_dist:.0f}" if puck_enemy_dist else ""
            cv2.putText(frame, dist_label, (enemy_x + 12, enemy_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"PWM: {last_sent_pwm:+d}", (w - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    ser.close()