from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import ExtendedKalmanFilter
from scipy.ndimage import binary_dilation

# --- GRID CONFIGURATIONS (10x10m Arenas) --- #
RESOLUTION = 10 
OFFSET = 5        
MAP_SIZE = 100   

robot = Robot()
timestep = int(robot.getBasicTimeStep())
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Hardware setup
gps = robot.getDevice('gps'); gps.enable(timestep)
imu = robot.getDevice('inertial unit'); imu.enable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX'); lidar.enable(timestep)
motors = [robot.getDevice(n) for n in ['back left wheel', 'back right wheel', 'front left wheel', 'front right wheel']]

for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Initialize Extended Kalman Filter for stable localization
ekf = ExtendedKalmanFilter(dt=timestep/1000)
grid_prob = np.zeros((MAP_SIZE, MAP_SIZE))

def world_to_grid(x, y):
    gx = int(round((x + OFFSET) * RESOLUTION))
    gy = int(round((y + OFFSET) * RESOLUTION))
    return gx, gy

plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))

print(">>> AMR MAPPER Running")
print(">>> Tip: Use the arrow keys in Webots to move the robot.")

# --- DEFINITIVE SLAM SCRIPT ---
last_rotation_time = 0
while robot.step(timestep) != -1:
    current_time = robot.getTime()
    pos = gps.getValues()
    theta = imu.getRollPitchYaw()[2] # Yaw (orientation)
    scan = lidar.getRangeImage()
    
    if np.isnan(pos[0]) or scan is None: continue

    xf, yf = pos[0], pos[1]
    
    # KEYBOARD CONTROL
    key = keyboard.getKey()
    v_l, v_r = 0, 0
    speed = 2.0
    
    is_rotating = (key == keyboard.LEFT or key == keyboard.RIGHT)
    if key == keyboard.UP: v_l, v_r = speed, speed
    elif key == keyboard.DOWN: v_l, v_r = -speed, -speed
    elif key == keyboard.LEFT: v_l, v_r = -speed, speed
    elif key == keyboard.RIGHT: v_l, v_r = speed, -speed

    if is_rotating:
        last_rotation_time = current_time

    # MAPPING WITH AXIS CORRECTION
    if not is_rotating and (current_time - last_rotation_time) > 0.6:
        n = len(scan)
        fov = np.radians(240)
        step = fov / (n - 1)

        for i in range(0, n, 4):
            dist = scan[i]
            if 0.5 < dist < 5:
                # Correcting laser angle mapping
                angle_laser = (fov / 2) - (i * step) 
                angle_world = theta + angle_laser
                
                # Projected coordinates
                ox = xf + dist * np.cos(angle_world)
                oy = yf + dist * np.sin(angle_world)
                
                gx, gy = world_to_grid(ox, oy)
                if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                    # Slow accumulation to filter noise
                    grid_prob[gy, gx] += 0.1

    # CLEANUP AND BINARIZATION
    grx, gry = world_to_grid(xf, yf)
    grid_prob[max(0, gry-2):gry+3, max(0, grx-2):grx+3] = 0.0 # Clear robot footprint
    grid_prob = np.clip(grid_prob, 0, 1)

    # MOTOR CONTROL
    motors[0].setVelocity(v_l); motors[2].setVelocity(v_l)
    motors[1].setVelocity(v_r); motors[3].setVelocity(v_r)

    # REAL-TIME PLOTTING
    if int(current_time * 10) % 10 == 0:
        ax.clear()
        
        # Binarization: Walls only if confidence > 0.85
        final_map = (grid_prob > 0.85).astype(float)
        
        # Dilation: Expand walls for A* safety
        safe_map = binary_dilation(final_map).astype(float) 
        
        ax.imshow(safe_map, origin='lower', cmap='binary', extent=[-5, 5, -5, 5])
        ax.plot(xf, yf, 'ro') 
        ax.set_title(f"Final SLAM - Time: {current_time:.1f}s")
        plt.pause(0.001)

        # Save map for A* planner
        np.save('mapa_A_star.npy', safe_map)
