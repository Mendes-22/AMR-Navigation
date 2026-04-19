from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import ExtendedKalmanFilter
from path_planner import A_star, smooth_path, world_to_grid, grid_to_world

# --- CONFIGURATIONS ---
RESOLUTION = 10; OFFSET = 5; MAX_SPEED = 6.4
TARGET_POSITION = (2.5, 2.0) 

robot = Robot()
timestep = int(robot.getBasicTimeStep())
gps = robot.getDevice('gps'); gps.enable(timestep)
imu = robot.getDevice('inertial unit'); imu.enable(timestep)
motors = [robot.getDevice(n) for n in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']]
for m in motors: m.setPosition(float('inf')); m.setVelocity(0.0)

# Load SLAM Map
try:
    grid_real = np.load('mapa_final_a_star.npy')
    print(">>> SLAM map loaded!")
except:
    print(">>> ERROR: 'mapa_final_a_star.npy' not found!")
    grid_real = np.zeros((100, 100))

ekf = ExtendedKalmanFilter(dt=timestep/1000)
path = []; point_idx = 0; v, w = 0.0, 0.0
is_calculated = False

print(f"🛰️ AMR Target Position: {TARGET_POSITION}")

while robot.step(timestep) != -1:
    pos_current = gps.getValues()
    theta_current = imu.getRollPitchYaw()[2]
    
    if np.isnan(pos_current[0]): continue

    # EKF Localization
    ekf.predict(v, w)
    ekf.update(np.array([[pos_current[0]], [pos_current[1]], [theta_current]]))
    xf, yf, tf = ekf.x.flatten()

    # AUTOMATIC PATH PLANNING + SHOWCASE PLOT
    if not is_calculated:
        print(f">>> AMR Start position: ({xf:.2f}, {yf:.2f})")
        start_node = world_to_grid(xf, yf)
        goal_node = world_to_grid(TARGET_POSITION[0], TARGET_POSITION[1])
        
        # Calculate route using A*
        raw_path = A_star(grid_real, start_node, goal_node)
        
        if raw_path:
            smooth_route = smooth_path(raw_path, grid_real)
            path = [grid_to_world(p[0], p[1]) for p in smooth_route]
            is_calculated = True
            
            # --- SHOWCASE PLOT ---
            plt.figure(figsize=(7, 7))
            plt.imshow(grid_real, origin='lower', cmap='binary', extent=[-5, 5, -5, 5])
            cx, cy = zip(*path)
            plt.plot(cx, cy, 'r-o', markersize=4, label='Planned Path')
            plt.plot(xf, yf, 'go', label='START')
            plt.plot(TARGET_POSITION[0], TARGET_POSITION[1], 'bx', label='TARGET')
            plt.title(f"A* path planning: {len(path)} points")
            plt.legend()
            plt.show() 
        else:
            print("❌ ERROR: Target not reachable!"); break

    # NAVIGATION
    if is_calculated:
        # Check if there are points left to follow
        if point_idx < len(path):
            target = path[point_idx]
            dx, dy = target[0] - xf, target[1] - yf
            dist_to_point = np.sqrt(dx**2 + dy**2)
            dist_to_final = np.sqrt((TARGET_POSITION[0]-xf)**2 + (TARGET_POSITION[1]-yf)**2)
            
            # --- STOP CONDITION ---
            if dist_to_final < 0.05 or (point_idx == len(path)-1 and dist_to_point < 0.1):
                for m in motors: 
                    m.setVelocity(0.0)
                break 

            # Angle error calculation
            target_angle = np.arctan2(dy, dx)
            err_a = (target_angle - tf + np.pi) % (2 * np.pi) - np.pi
            
            # Velocity control proportional to final distance
            if abs(err_a) > 0.4:
                v, w = 0.0, np.clip(err_a * 4.0, -2.5, 2.5)
            else:
                # Smooth slowdown when approaching the target
                v = np.clip(dist_to_final * 2.0, 0.8, 3.8) 
                w = np.clip(err_a * 3.5, -1.5, 1.5)

            # Switch to next point with a 15cm margin
            if dist_to_point < 0.15: 
                point_idx += 1
        else:
            # Extra safety: stop if the path list ends
            for m in motors: 
                m.setVelocity(0.0)
            print(f"🏁 TARGET POSITION REACHED: ({xf:.2f}, {yf:.2f})")
            break

    # APPLY MOTORS
    vl, vr = np.clip(v-w, -MAX_SPEED, MAX_SPEED), np.clip(v+w, -MAX_SPEED, MAX_SPEED)
    motors[0].setVelocity(vl); motors[2].setVelocity(vl)
    motors[1].setVelocity(vr); motors[3].setVelocity(vr)
