import os
import sys

# Headless SDL mode if imported
if __name__ != "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import json
import time
import math
import numpy as np

# Initialize Pygame
pygame.init()
pygame.font.init()

# Window setup constants
WIDTH, HEIGHT = 500, 400

# Global screen variable
screen = None

if __name__ == "__main__":
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("FogVision ADAS Simulator")

# Colors
COLOR_BG = (10, 17, 40)       # Royal Midnight Blue
COLOR_PANEL = (16, 31, 66)    # Lighter Panel Blue
COLOR_TEXT_MUTED = (148, 163, 184)
COLOR_TEXT_LIGHT = (243, 244, 246)
COLOR_TEAL = (6, 182, 212)
COLOR_SAFE = (16, 185, 129)
COLOR_WARN = (245, 158, 11)
COLOR_DANGER = (239, 68, 68)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (253, 224, 71)

# Fonts
font_sm = pygame.font.SysFont("Arial", 14)
font_md = pygame.font.SysFont("Arial", 18, bold=True)
font_lg = pygame.font.SysFont("Arial", 28, bold=True)
font_speed = pygame.font.SysFont("Arial", 36, bold=True)

# State File Path
STATE_FILE = "sim_state.json"

# Simulation constants
CAR_WIDTH, CAR_HEIGHT = 40, 70
CAR_X = WIDTH // 2 - CAR_WIDTH // 2
CAR_Y = HEIGHT - 220

# Animation variables
road_offset = 0.0

def load_sim_state():
    """Load simulator state from JSON file."""
    if not os.path.exists(STATE_FILE):
        # Default state if file doesn't exist
        return {
            "running": False,
            "speed": 50.0,
            "braking": False,
            "alerts": [],
            "esp32_sensor_data": {"left": 80.0, "middle": 80.0, "right": 80.0},
            "last_update": time.time()
        }
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        # Fallback on parse errors
        return {
            "running": False,
            "speed": 50.0,
            "braking": False,
            "alerts": [],
            "esp32_sensor_data": {"left": 80.0, "middle": 80.0, "right": 80.0},
            "last_update": time.time()
        }

def save_sim_state(state):
    """Save simulator state to JSON file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

def draw_speedometer(surface, speed, is_braking):
    """Draw a modern circular arc speedometer."""
    center_x, center_y = 100, HEIGHT - 75
    radius = 50
    start_angle = math.pi * 0.85
    end_angle = math.pi * 2.15
    
    # Draw arc background
    pygame.draw.arc(surface, COLOR_PANEL, (center_x - radius, center_y - radius, radius * 2, radius * 2), -end_angle, -start_angle, 6)
    
    # Use absolute speed for speedometer dial rendering
    abs_speed = abs(speed)
    speed_pct = min(abs_speed / 120.0, 1.0)
    current_angle = start_angle + speed_pct * (end_angle - start_angle)
    
    color = COLOR_DANGER if is_braking else COLOR_TEAL
    if speed_pct > 0:
        pygame.draw.arc(surface, color, (center_x - radius, center_y - radius, radius * 2, radius * 2), -current_angle, -start_angle, 6)
    
    # Speed numbers
    speed_txt = font_speed.render(f"{int(abs_speed)}", True, COLOR_TEXT_LIGHT)
    unit_txt = font_sm.render("km/h", True, COLOR_TEXT_MUTED)
    
    surface.blit(speed_txt, (center_x - speed_txt.get_width() // 2, center_y - 20))
    surface.blit(unit_txt, (center_x - unit_txt.get_width() // 2, center_y + 15))

def render_simulation_frame(speed, is_braking, alerts, sensors, is_active, turn=0, dt=1.0/30.0):
    global road_offset
    
    # Create surface
    surface = pygame.Surface((WIDTH, HEIGHT))
    
    # Calculate current_speed animation (can just use speed directly)
    # Update road offset animation
    speed_factor = speed / 10.0
    road_offset = (road_offset + speed_factor) % 40
    
    # Clear surface
    surface.fill(COLOR_BG)
    
    # --- DRAW ROAD PANEL (Middle Section) ---
    road_x_left = 180
    road_x_right = WIDTH - 180
    road_width = road_x_right - road_x_left
    
    # Fill grass sides
    pygame.draw.rect(surface, (16, 26, 52), (0, 0, road_x_left, HEIGHT))
    pygame.draw.rect(surface, (16, 26, 52), (road_x_right, 0, WIDTH - road_x_right, HEIGHT))
    
    # Draw road base
    pygame.draw.rect(surface, (30, 41, 59), (road_x_left, 0, road_width, HEIGHT))
    
    # Draw solid side lines
    pygame.draw.line(surface, COLOR_WHITE, (road_x_left, 0), (road_x_left, HEIGHT), 4)
    pygame.draw.line(surface, COLOR_WHITE, (road_x_right, 0), (road_x_right, HEIGHT), 4)
    
    # Draw animated center dashed line
    line_length = 20
    gap_length = 20
    num_lines = HEIGHT // (line_length + gap_length) + 2
    for i in range(num_lines):
        y_pos = -40 + i * (line_length + gap_length) + road_offset
        pygame.draw.line(surface, COLOR_YELLOW, (WIDTH // 2, y_pos), (WIDTH // 2, y_pos + line_length), 3)
        
    # Displace the car's horizontal position based on the steering turn rate
    # turn ranges from -100 to 100. Let's scale it to shift up to 35 pixels left or right.
    car_x_offset = int((turn / 100.0) * 35.0)
    sim_car_x = CAR_X + car_x_offset

    # Draw vehicle
    car_color = (194, 65, 12) if is_braking else COLOR_TEAL
    pygame.draw.rect(surface, car_color, (sim_car_x, CAR_Y, CAR_WIDTH, CAR_HEIGHT), border_radius=5)
    # Windows
    pygame.draw.rect(surface, (30, 30, 30), (sim_car_x + 5, CAR_Y + 12, CAR_WIDTH - 10, 15), border_radius=2)
    pygame.draw.rect(surface, (30, 30, 30), (sim_car_x + 5, CAR_Y + 45, CAR_WIDTH - 10, 10), border_radius=1)
    # Headlights
    pygame.draw.circle(surface, (254, 240, 138), (sim_car_x + 8, CAR_Y + 3), 4)
    pygame.draw.circle(surface, (254, 240, 138), (sim_car_x + CAR_WIDTH - 8, CAR_Y + 3), 4)
    # Taillights
    taillight_color = (255, 0, 0) if is_braking else (180, 0, 0)
    pygame.draw.rect(surface, taillight_color, (sim_car_x + 4, CAR_Y + CAR_HEIGHT - 5, 8, 4))
    pygame.draw.rect(surface, taillight_color, (sim_car_x + CAR_WIDTH - 12, CAR_Y + CAR_HEIGHT - 5, 8, 4))
    
    # --- DRAW HUD OVERLAYS ---
    pygame.draw.rect(surface, COLOR_PANEL, (0, 0, WIDTH, 35))
    pygame.draw.line(surface, COLOR_TEAL, (0, 35), (WIDTH, 35), 2)
    
    title_txt = font_md.render("FOGVISION ADAS COGNITIVE SIMULATOR", True, COLOR_TEAL)
    surface.blit(title_txt, (15, 8))
    
    status_color = COLOR_SAFE if is_active else COLOR_TEXT_MUTED
    status_lbl = "ACTIVE" if is_active else "STANDBY"
    status_txt = font_sm.render(f"SYS: {status_lbl}", True, status_color)
    surface.blit(status_txt, (WIDTH - status_txt.get_width() - 15, 10))
    
    pygame.draw.rect(surface, COLOR_PANEL, (0, HEIGHT - 130, WIDTH, 130))
    pygame.draw.line(surface, COLOR_TEAL, (0, HEIGHT - 130), (WIDTH, HEIGHT - 130), 2)
    
    draw_speedometer(surface, speed, is_braking)
    
    alert_start_y = HEIGHT - 110
    if is_braking:
        brake_txt = font_md.render("🚨 ABS AUTOMATED BRAKE ACTIVE", True, COLOR_DANGER)
        surface.blit(brake_txt, (180, alert_start_y))
        alert_start_y += 22
        
    if alerts:
        first_alert = alerts[0].get("message", str(alerts[0])) if isinstance(alerts[0], dict) else str(alerts[0])
        clean_alert = first_alert.replace("🚨", "").replace("⚠️", "").strip()
        alert_color = COLOR_DANGER if "CRITICAL" in clean_alert or "Collision" in clean_alert else COLOR_WARN
        alert_txt = font_sm.render(clean_alert[:40], True, alert_color)
        surface.blit(alert_txt, (180, alert_start_y))
    else:
        if not is_braking:
            ok_txt = font_sm.render("✅ ADAS SAFETY: ROAD CONDITIONS NOMINAL", True, COLOR_SAFE)
            surface.blit(ok_txt, (180, alert_start_y))
            
    sensor_start_x = WIDTH - 140
    pygame.draw.line(surface, (30, 41, 59), (sensor_start_x - 15, HEIGHT - 120), (sensor_start_x - 15, HEIGHT - 10), 1)
    
    lbl_sensors = font_md.render("PLAN A SENSORS", True, COLOR_TEAL)
    surface.blit(lbl_sensors, (sensor_start_x, HEIGHT - 115))
    
    left_dist = sensors.get("left")
    left_dist_val = f"{left_dist:.2f}m" if isinstance(left_dist, (int, float)) and left_dist < 80 else "INF"
    left_color = COLOR_DANGER if isinstance(left_dist, (int, float)) and left_dist < 0.3 else COLOR_WARN if isinstance(left_dist, (int, float)) and left_dist < 0.6 else COLOR_TEXT_LIGHT
    left_txt = font_sm.render(f"Left: {left_dist_val}", True, left_color)
    surface.blit(left_txt, (sensor_start_x, HEIGHT - 90))
    
    mid_dist = sensors.get("middle")
    mid_dist_val = f"{mid_dist:.2f}m" if isinstance(mid_dist, (int, float)) and mid_dist < 80 else "INF"
    mid_color = COLOR_DANGER if isinstance(mid_dist, (int, float)) and mid_dist < 0.3 else COLOR_WARN if isinstance(mid_dist, (int, float)) and mid_dist < 0.6 else COLOR_TEXT_LIGHT
    mid_txt = font_sm.render(f"Middle: {mid_dist_val}", True, mid_color)
    surface.blit(mid_txt, (sensor_start_x, HEIGHT - 70))
    
    right_dist = sensors.get("right")
    right_dist_val = f"{right_dist:.2f}m" if isinstance(right_dist, (int, float)) and right_dist < 80 else "INF"
    right_color = COLOR_DANGER if isinstance(right_dist, (int, float)) and right_dist < 0.3 else COLOR_WARN if isinstance(right_dist, (int, float)) and right_dist < 0.6 else COLOR_TEXT_LIGHT
    right_txt = font_sm.render(f"Right: {right_dist_val}", True, right_color)
    surface.blit(right_txt, (sensor_start_x, HEIGHT - 50))
    
    img_data = pygame.image.tostring(surface, "RGB")
    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
    return img_array

def main():
    global road_offset
    clock = pygame.time.Clock()
    
    # Initialize animation state
    current_speed = 0.0
    
    # Manual control variables
    car_x = CAR_X
    user_target_speed = 50.0
    manual_control_active = False
    last_keyboard_input_time = 0.0
    last_esp32_sent_time = 0.0
    last_esp32_sent_speed = None
    last_esp32_sent_turn = None
    
    running_sim = True
    while running_sim:
        # Event handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_sim = False
        
        # Load state from file
        state = load_sim_state()
        target_speed = float(state.get("speed", 50.0))
        is_braking = bool(state.get("braking", False))
        alerts = state.get("alerts", [])
        sensors = state.get("esp32_sensor_data", {"left": 80.0, "middle": 80.0, "right": 80.0})
        is_system_active = bool(state.get("running", False))
        esp32_ip = state.get("esp32_ip", "")
        
        # Calculate max safe speed limit based on estimated fog density:
        fog_density = float(state.get("fog_density", 0.0))
        if fog_density >= 80.0:
            max_safe_speed = 30.0
        elif fog_density >= 60.0:
            max_safe_speed = 50.0
        elif fog_density >= 40.0:
            max_safe_speed = 70.0
        else:
            max_safe_speed = 100.0

        # Near range object check to block movement
        near_obstacle = is_braking
        critical_limit = 0.3 if is_system_active else 10.0
        left_dist = sensors.get("left", 80.0)
        mid_dist = sensors.get("middle", 80.0)
        right_dist = sensors.get("right", 80.0)
        if (left_dist < critical_limit) or (mid_dist < critical_limit) or (right_dist < critical_limit):
            near_obstacle = True

        if near_obstacle:
            user_target_speed = 0.0
            current_speed = 0.0
            is_braking = True  # force braking flag so taillights stay red
            
        # Keyboard steering and acceleration
        keys = pygame.key.get_pressed()
        arrow_key_pressed = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]
        if arrow_key_pressed:
            if not manual_control_active:
                user_target_speed = current_speed
            manual_control_active = True
            last_keyboard_input_time = time.time()
            
            # Steering LEFT / RIGHT
            road_x_left = 180
            road_x_right = WIDTH - 180
            if keys[pygame.K_LEFT]:
                car_x = max(road_x_left + 5, car_x - 4)
            elif keys[pygame.K_RIGHT]:
                car_x = min(road_x_right - CAR_WIDTH - 5, car_x + 4)
                
            # Speed UP / DOWN - block speed up if near obstacle is present
            if keys[pygame.K_UP] and not near_obstacle:
                user_target_speed = min(max_safe_speed, user_target_speed + 1.5)
            elif keys[pygame.K_DOWN]:
                user_target_speed = max(0.0, user_target_speed - 2.0)
        else:
            # Self-centering if manual control is active but no steer key is pressed
            if manual_control_active:
                if car_x < CAR_X:
                    car_x = min(CAR_X, car_x + 1)
                elif car_x > CAR_X:
                    car_x = max(CAR_X, car_x - 1)
                    
        # Automatic timeout to revert back to telemetry control
        if manual_control_active and (time.time() - last_keyboard_input_time > 4.0):
            manual_control_active = False
            
        # Revert car_x and speed targets if manual control is not active
        if not manual_control_active:
            if car_x < CAR_X:
                car_x = min(CAR_X, car_x + 2)
            elif car_x > CAR_X:
                car_x = max(CAR_X, car_x - 2)
                
        # Enforce safe speed overrides
        if user_target_speed > max_safe_speed:
            user_target_speed = max_safe_speed
            
        # Determine speed target
        speed_to_target = user_target_speed if manual_control_active else target_speed
        if not manual_control_active:
            user_target_speed = target_speed  # keep in sync
            
        # Smoothly interpolate speed
        if is_braking:
            # Rapid deceleration on automated braking
            current_speed -= 5.0
            if current_speed < 0.0:
                current_speed = 0.0
        else:
            # Smooth transition to target speed
            if current_speed < speed_to_target:
                current_speed += 1.5
                if current_speed > speed_to_target:
                    current_speed = speed_to_target
            elif current_speed > speed_to_target:
                current_speed -= 2.0
                if current_speed < speed_to_target:
                    current_speed = speed_to_target
                    
        # Update road offset animation
        speed_factor = current_speed / 10.0
        road_offset = (road_offset + speed_factor) % 40
        
        # Write back manual control state to sim_state.json if active
        if manual_control_active:
            state["speed"] = current_speed
            # Map car_x (range 185 to 275, center CAR_X = 230) to turn (-100 to 100)
            turn_value = int(((car_x - CAR_X) / 45.0) * 100.0)
            state["turn"] = turn_value
            state["manual_control"] = True
            save_sim_state(state)
            
            # Send motor signals asynchronously to ESP32 from simulator if manually controlled
            if esp32_ip:
                # Apply correct sign for speed based on K_DOWN
                speed_val = int(current_speed)
                if keys[pygame.K_DOWN]:
                    speed_val = -speed_val
                
                current_sent_time = time.time()
                is_stopped_now = (speed_val == 0 and turn_value == 0)
                was_stopped_before = (last_esp32_sent_speed == 0 and last_esp32_sent_turn == 0)
                should_send = not (is_stopped_now and was_stopped_before)

                if should_send:
                    time_elapsed = current_sent_time - last_esp32_sent_time
                    value_changed = (last_esp32_sent_speed != speed_val or 
                                     last_esp32_sent_turn != turn_value)
                    
                    if last_esp32_sent_speed is None or value_changed or time_elapsed > 0.08:
                        last_esp32_sent_time = current_sent_time
                        last_esp32_sent_speed = speed_val
                        last_esp32_sent_turn = turn_value
                        
                        def send_esp32_control_sim(ip, s, t):
                            import esp32_module
                            esp32_module.send_motor_speed(ip, s, t, timeout=0.30)
                        
                        import threading
                        threading.Thread(
                            target=send_esp32_control_sim, 
                            args=(esp32_ip, speed_val, turn_value), 
                            daemon=True
                        ).start()
        elif state.get("manual_control", False):
            # Reset manual control flag once when returning to auto control
            state["manual_control"] = False
            save_sim_state(state)
        
        # Clear screen
        screen.fill(COLOR_BG)
        
        # --- DRAW ROAD PANEL (Middle Section) ---
        road_x_left = 180
        road_x_right = WIDTH - 180
        road_width = road_x_right - road_x_left
        
        # Fill grass sides
        pygame.draw.rect(screen, (16, 26, 52), (0, 0, road_x_left, HEIGHT))
        pygame.draw.rect(screen, (16, 26, 52), (road_x_right, 0, WIDTH - road_x_right, HEIGHT))
        
        # Draw road base
        pygame.draw.rect(screen, (30, 41, 59), (road_x_left, 0, road_width, HEIGHT))
        
        # Draw solid side lines
        pygame.draw.line(screen, COLOR_WHITE, (road_x_left, 0), (road_x_left, HEIGHT), 4)
        pygame.draw.line(screen, COLOR_WHITE, (road_x_right, 0), (road_x_right, HEIGHT), 4)
        
        # Draw animated center dashed line
        line_length = 20
        gap_length = 20
        num_lines = HEIGHT // (line_length + gap_length) + 2
        for i in range(num_lines):
            y_pos = -40 + i * (line_length + gap_length) + road_offset
            pygame.draw.line(screen, COLOR_YELLOW, (WIDTH // 2, y_pos), (WIDTH // 2, y_pos + line_length), 3)
            
        # Draw vehicle
        # Car body
        car_color = (194, 65, 12) if is_braking else COLOR_TEAL
        pygame.draw.rect(screen, car_color, (car_x, CAR_Y, CAR_WIDTH, CAR_HEIGHT), border_radius=5)
        # Windows
        pygame.draw.rect(screen, (30, 30, 30), (car_x + 5, CAR_Y + 12, CAR_WIDTH - 10, 15), border_radius=2)
        pygame.draw.rect(screen, (30, 30, 30), (car_x + 5, CAR_Y + 45, CAR_WIDTH - 10, 10), border_radius=1)
        # Headlights (white yellow)
        pygame.draw.circle(screen, (254, 240, 138), (car_x + 8, CAR_Y + 3), 4)
        pygame.draw.circle(screen, (254, 240, 138), (car_x + CAR_WIDTH - 8, CAR_Y + 3), 4)
        # Taillights (red)
        taillight_color = (255, 0, 0) if is_braking else (180, 0, 0)
        pygame.draw.rect(screen, taillight_color, (car_x + 4, CAR_Y + CAR_HEIGHT - 5, 8, 4))
        pygame.draw.rect(screen, taillight_color, (car_x + CAR_WIDTH - 12, CAR_Y + CAR_HEIGHT - 5, 8, 4))
        
        # --- DRAW HEADS-UP DISPLAY (HUD) OVERLAYS ---
        # Top Header Bar
        pygame.draw.rect(screen, COLOR_PANEL, (0, 0, WIDTH, 35))
        pygame.draw.line(screen, COLOR_TEAL, (0, 35), (WIDTH, 35), 2)
        
        title_txt = font_md.render("FOGVISION ADAS COGNITIVE SIMULATOR", True, COLOR_TEAL)
        screen.blit(title_txt, (15, 8))
        
        # Active Status Indicator
        if manual_control_active:
            status_color = COLOR_WARN
            status_lbl = "MANUAL OVERRIDE"
        else:
            status_color = COLOR_SAFE if is_system_active else COLOR_TEXT_MUTED
            status_lbl = "ACTIVE" if is_system_active else "STANDBY"
        status_txt = font_sm.render(f"SYS: {status_lbl}", True, status_color)
        screen.blit(status_txt, (WIDTH - status_txt.get_width() - 15, 10))
        
        # Bottom HUD Panel
        pygame.draw.rect(screen, COLOR_PANEL, (0, HEIGHT - 130, WIDTH, 130))
        pygame.draw.line(screen, COLOR_TEAL, (0, HEIGHT - 130), (WIDTH, HEIGHT - 130), 2)
        
        # Left HUD: Speedometer
        draw_speedometer(screen, current_speed, is_braking)
        
        # Middle HUD: Braking and Alerts
        alert_start_y = HEIGHT - 110
        if near_obstacle:
            block_txt = font_md.render("🚨 CRITICAL BLOCK: COLLISION THREAT - MOVEMENT BLOCKED", True, COLOR_DANGER)
            screen.blit(block_txt, (180, alert_start_y))
            alert_start_y += 22
        elif is_braking:
            brake_txt = font_md.render("🚨 ABS AUTOMATED BRAKE ACTIVE", True, COLOR_DANGER)
            screen.blit(brake_txt, (180, alert_start_y))
            alert_start_y += 22
            
        if alerts:
            # Draw top alert
            first_alert = alerts[0]
            # Strip emojis for pygame compatibility if any
            clean_alert = first_alert.replace("🚨", "").replace("⚠️", "").strip()
            alert_color = COLOR_DANGER if "CRITICAL" in clean_alert or "Collision" in clean_alert else COLOR_WARN
            alert_txt = font_sm.render(clean_alert[:40], True, alert_color)
            screen.blit(alert_txt, (180, alert_start_y))
        else:
            if not near_obstacle and not is_braking:
                ok_txt = font_sm.render("✅ ADAS SAFETY: ROAD CONDITIONS NOMINAL", True, COLOR_SAFE)
                screen.blit(ok_txt, (180, alert_start_y))
                
        # Right HUD: Plan A Sensor Statuses
        sensor_start_x = WIDTH - 140
        pygame.draw.line(screen, (30, 41, 59), (sensor_start_x - 15, HEIGHT - 120), (sensor_start_x - 15, HEIGHT - 10), 1)
        
        lbl_sensors = font_md.render("PLAN A SENSORS", True, COLOR_TEAL)
        screen.blit(lbl_sensors, (sensor_start_x, HEIGHT - 115))
        
        left_dist = sensors.get("left")
        left_dist_val = f"{left_dist:.1f}m" if isinstance(left_dist, (int, float)) and left_dist < 80 else "INF"
        left_color = COLOR_DANGER if isinstance(left_dist, (int, float)) and left_dist < 15 else COLOR_TEXT_LIGHT
        left_txt = font_sm.render(f"Left: {left_dist_val}", True, left_color)
        screen.blit(left_txt, (sensor_start_x, HEIGHT - 90))
        
        mid_dist = sensors.get("middle")
        mid_dist_val = f"{mid_dist:.1f}m" if isinstance(mid_dist, (int, float)) and mid_dist < 80 else "INF"
        mid_color = COLOR_DANGER if isinstance(mid_dist, (int, float)) and mid_dist < 15 else COLOR_TEXT_LIGHT
        mid_txt = font_sm.render(f"Middle: {mid_dist_val}", True, mid_color)
        screen.blit(mid_txt, (sensor_start_x, HEIGHT - 70))
        
        right_dist = sensors.get("right")
        right_dist_val = f"{right_dist:.1f}m" if isinstance(right_dist, (int, float)) and right_dist < 80 else "INF"
        right_color = COLOR_DANGER if isinstance(right_dist, (int, float)) and right_dist < 15 else COLOR_TEXT_LIGHT
        right_txt = font_sm.render(f"Right: {right_dist_val}", True, right_color)
        screen.blit(right_txt, (sensor_start_x, HEIGHT - 50))
        
        # Render and update frame
        pygame.display.flip()
        clock.tick(30) # 30 FPS for simulation smooth motion
        
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
