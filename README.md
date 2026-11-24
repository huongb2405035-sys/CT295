# Target-Following Car (ESP32 + Phone Camera + MediaPipe)

The car uses a phone camera to detect a **hand or a human body** with MediaPipe, then turns and drives to follow the target. The car uses only **one L298N module**.

## Architecture

```
Phone (IP Webcam) --MJPEG video--> Computer: phone_tracker.py (MediaPipe Hand + Pose)
                                          |  UDP "D,det,err,area"
                                          v
                                    ESP32: esp32_car.ino (following logic + motor control)
                                          |  UDP reports (ST,... / TARGET_FOUND / TARGET_LOCKED ...)
                                          v
                                    Computer (printed on screen)

Browser (index.html served by ESP32) --WebSocket "NORMAL" / "SEARCHING"--> ESP32 (start/stop the car)
```

The ESP32 is not powerful enough to run the detection model, so MediaPipe runs on the computer. The ESP32 handles all the following logic, PWM motor control, and the control web page.

## Detection (phone_tracker.py)

- **Hand**: `HandLandmarker` (`hand_landmarker.task`), up to 2 hands.
- **Body**: `PoseLandmarker` (`pose_landmarker_lite.task`). A person counts as detected when at least 2 of the 4 torso points (left/right shoulder, left/right hip) have visibility >= 0.5.
- The target is confirmed after `TARGET_REQUIRED` (5) consecutive frames containing a hand or a person, and dropped after `NO_TARGET_REQUIRED` (10) consecutive frames without one.
- The **body** bounding box takes priority; if there is no body, the bounding box of the first **hand** is used.
- Two values are computed from the bounding box and sent to the ESP32:
  - `err`: horizontal offset of the target's center from the frame center, from -1 (far left) to +1 (far right).
  - `area`: ratio of bounding box area to frame area (0..1), used to estimate distance (larger = closer).
- UDP packet sent to the ESP32: `D,<det>,<err>,<area>`. Pressing `r` sends `R` to reset.
- The computer finds the ESP32 on the network by broadcasting `HELLO`; the ESP32 replies with `ST,...` so the computer learns its IP. To set it manually, fill in `ESP32_IP`.

## Control Logic (esp32_car.ino)

The car has 2 modes, switched from the web page: `NORMAL` (stands still, default at startup) and `SEARCHING` (automatically follows the target).

In `SEARCHING` mode, the car has 3 states:

| State | Condition | Action |
|---|---|---|
| `SEARCH` | Target not seen | Spins in place in short pulses (preferring the side where the target was last seen), pausing between pulses so the camera can detect |
| `TRACK` | Target seen | Large offset (`|err| > PIVOT_ERR`): pivots in place to realign. Small offset: drives forward while steering proportionally to `err`. Slows down when close |
| `LOCKED` | `area >= STOP_AREA` | Stops and sends `TARGET_LOCKED`. If the target shifts sideways, rotates to follow. When `area < RESUME_AREA` (target moves away), follows again automatically, no reset needed |

Messages the ESP32 sends back to the computer: `TARGET_FOUND`, `TARGET_LOCKED`, `TARGET_MOVING`, `RESET_OK`, and a status packet `ST,<NORMAL|SEARCHING>,<state>,err=...,area=...` (every 500 ms).

Safety in the code:
- Packets older than `FRESH_MS` (250 ms) are treated as lost signal and the car stops.
- If the target is lost for longer than `LOST_GRACE_MS` (600 ms), the car returns to the `SEARCH` state.
- In `NORMAL` mode, all motor commands are blocked.

## Running Steps

Connect all devices to the same network. Suggestion: turn on the phone's hotspot, connect the computer to the phone's WiFi, and upload the code from the computer to the ESP32.

1. **Phone**: install the *IP Webcam* app (Android), tap Start server, and note the address in the form `http://192.168.x.x:8080/video`.
2. **ESP32** (Arduino IDE, requires **ESP32 Arduino core 3.x** because it uses `analogWrite`):
   - Install the *AsyncTCP* and *ESPAsyncWebServer* libraries. ESP32Servo is not needed.
   - Open `esp32_car.ino`, fill in `WIFI_SSID` / `WIFI_PASS`, and upload the code.
   - Upload `index.html` to the ESP32's LittleFS storage (the control page sends `NORMAL` / `SEARCHING` over WebSocket).
   - Open the Serial Monitor (115200) to see the ESP32's IP. The control page is at `http://<ESP32 IP>` or `http://xebam.local`.
3. **Computer** (same WiFi):
   - Install the libraries: `pip install mediapipe opencv-python numpy`
   - Download the two model files `hand_landmarker.task` and `pose_landmarker_lite.task` from the MediaPipe site and place them in the same folder as `phone_tracker.py`.
   - Change `CAMERA_URL` in `phone_tracker.py` to the phone camera address, then run `python phone_tracker.py`.
4. Open the control page and press **SEARCHING** to make the car start following; press **NORMAL** to stop it.
5. Keys in the video window: `r` = reset the car, `q` = quit.

## Wiring (1 L298N)

| Device | Pins |
|---|---|
| L298N IN1, IN2 (**left** motors, OUT1/OUT2) | GPIO 27, GPIO 26 |
| L298N IN3, IN4 (**right** motors, OUT3/OUT4) | GPIO 25, GPIO 33 |

- The 2 left motors are wired in parallel to OUT1/OUT2, and the 2 right motors in parallel to OUT3/OUT4.
- ENA, ENB: keep the jumpers on (always enabled) or connect them to 5V. Speed is controlled by PWM on the IN pins.
- Connect a common GND between the battery, the L298N, and the ESP32.
- If your wiring differs, edit the constants `PIN_L1`, `PIN_L2`, `PIN_R1`, `PIN_R2` at the top of the `.ino` file. If a wheel spins backwards, change `LEFT_SIGN` / `RIGHT_SIGN`.

## Calibration

- `STOP_AREA`, `RESUME_AREA`: `area` thresholds for stopping and for resuming following (check the `area` value on the video window when the target is at the desired distance). Keep `RESUME_AREA < STOP_AREA` so the car does not jitter.
- `DEADBAND`: the zone considered "centered"; if too small, the car will wobble left and right.
- `PIVOT_ERR`, `NEAR_TURN_ERR`: offset thresholds for pivoting in place while following / while standing close to the target.
- `BASE_SPEED`, `SLOW_SPEED`, `STEER_GAIN`, `TURN_SPEED`, `SEARCH_SPEED`: PWM speeds (0..255). `MIN_PWM`: below this the motors lack the torque to turn.
- `SEARCH_PULSE_MS`, `SEARCH_PAUSE_MS`: spin pulse length and wait time while searching for the target. Set `SEARCH_SPIN = false` to make the car wait in place instead of spinning to search.
- Computer side: `min_detection_confidence` / `min_tracking_confidence` (0.6), `TARGET_REQUIRED`, `NO_TARGET_REQUIRED`.
- If you enable image flipping (`cv2.flip`) in `phone_tracker.py`, you must invert the sign of `err`.

## Safety Notes

- **The car has no collision sensor.** It stops only based on `area` from the camera, so run it only in an open area with no obstacles and tune `STOP_AREA` appropriately.
- Lift the wheels off the ground for the first test run.
- The camera only sees the area in front; if the target leaves the frame, the car stops and then spins to search, and may turn into surrounding obstacles.
- Network/video latency can make the car react slowly; keep the ESP32, phone, and computer on a strong WiFi signal.
- Do not publish the `.ino` file publicly while it still contains your WiFi name and password.
