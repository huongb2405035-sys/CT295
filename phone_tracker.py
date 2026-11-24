import socket
import threading
import time
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# 1. CAMERA, MODEL, ESP32 CONFIG
# ==============================

# Replace with your phone camera address (usually http://)
CAMERA_URL = "http://10.166.82.170:8080/video"

# ESP32 IP: None = auto-discover on the network (recommended).
# Or set it manually, e.g. "10.166.82.198" (see the "ESP32 IP: ..." line in the Serial Monitor)
ESP32_IP = None
ESP32_PORT = 4210        # must match UDP_PORT in the .ino file
DISCOVER_TIMEOUT = 20    # seconds to wait while auto-discovering the ESP32

HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker_lite.task"

# ==============================
# 2. INITIALIZE MEDIAPIPE TASKS
# ==============================

hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

pose_base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# Indices of the 4 upper-body points in PoseLandmarker:
# 11: LEFT_SHOULDER, 12: RIGHT_SHOULDER, 23: LEFT_HIP, 24: RIGHT_HIP
IMPORTANT_POSE_INDICES = [11, 12, 23, 24]

# Get the skeleton connection list, compatible with different MediaPipe versions
POSE_CONNECTIONS = getattr(
    vision.PoseLandmarksConnections,
    "POSE_LANDMARKS",
    getattr(vision.PoseLandmarksConnections, "POSE_CONNECTIONS", None),
)


def draw_pose_safely(frame, landmarks):
    """Draw landmarks and skeleton lines (if available) without version-related errors."""
    try:
        if POSE_CONNECTIONS is not None:
            vision.drawing_utils.draw_landmarks(
                frame,
                landmarks,
                POSE_CONNECTIONS,
                vision.drawing_styles.get_default_pose_landmarks_style(),
            )
        else:
            vision.drawing_utils.draw_landmarks(frame, landmarks)
    except Exception:
        # Safe fallback: draw plain landmark dots if the style/connections have a different structure
        h, w = frame.shape[:2]
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)


def bbox_from_landmarks(landmarks, min_vis=None):
    """Bounding box (x0, y0, x1, y1) of the landmarks, normalized coordinates 0..1.
    min_vis: skip points with lower visibility (used for pose only)."""
    xs, ys = [], []
    for lm in landmarks:
        if min_vis is not None:
            vis = lm.visibility if lm.visibility is not None else 0.0
            if vis < min_vis:
                continue
        xs.append(lm.x)
        ys.append(lm.y)
    if not xs:
        return None
    x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
    y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
    return x0, y0, x1, y1


# ==============================
# 3. CAMERA CONNECTION + UDP SOCKET
# ==============================

class MjpegCapture:
    """Read the MJPEG stream (IP Webcam) directly over HTTP, without FFmpeg.
    A background thread always keeps the newest frame -> low latency. Used like cv2.VideoCapture."""

    def __init__(self, url):
        self.frame = None
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = True
        self.ok = False
        self.stream = None
        try:
            self.stream = urllib.request.urlopen(url, timeout=5)
            self.ok = True
        except Exception as e:
            print("Could not open stream:", e)
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()

    def isOpened(self):
        return self.ok

    def _loop(self):
        buf = b""
        while self.running:
            try:
                chunk = self.stream.read1(65536)
            except Exception:
                chunk = b""
            if not chunk:
                self.ok = False
                self.new_frame.set()     # wake up read() to report the lost connection
                return
            buf += chunk

            # Split out JPEG images (start FFD8, end FFD9), keep only the newest one
            last_jpg = None
            while True:
                a = buf.find(b"\xff\xd8")
                if a == -1:
                    buf = b""
                    break
                b = buf.find(b"\xff\xd9", a + 2)
                if b == -1:
                    buf = buf[a:]
                    break
                last_jpg = buf[a:b + 2]
                buf = buf[b + 2:]

            if last_jpg is not None:
                f = cv2.imdecode(np.frombuffer(last_jpg, np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    with self.lock:
                        self.frame = f
                    self.new_frame.set()

    def read(self):
        if not self.new_frame.wait(5.0):
            return False, None
        with self.lock:
            f = self.frame
            self.frame = None
            self.new_frame.clear()
        return f is not None, f

    def release(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.close()
            except Exception:
                pass


cap = MjpegCapture(CAMERA_URL)

if not cap.isOpened():
    print("Error: Could not connect to the camera!")
    exit()

print("Camera connected successfully!")

# UDP socket for sending data to the ESP32
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)


def local_broadcast_addr():
    """Broadcast address of the current network (assumes a /24 network), e.g. 192.168.1.255."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # sends no packet, only used to get the computer's IP
        ip = s.getsockname()[0]
        s.close()
        return ".".join(ip.split(".")[:3]) + ".255"
    except OSError:
        return None


def broadcast_hello():
    """Broadcast a HELLO packet so the ESP32 remembers the computer's address, then it sends ST,... back."""
    targets = ["255.255.255.255"]
    b = local_broadcast_addr()
    if b:
        targets.append(b)
    for t in targets:
        try:
            sock.sendto(b"HELLO", (t, ESP32_PORT))
        except OSError:
            pass


def discover_esp32(timeout):
    """Find the ESP32 on the network: broadcast HELLO, wait for an 'ST,...' reply, take the sender's IP."""
    print("Searching for ESP32 on the network...")
    sock.settimeout(1.0)
    t0 = time.time()
    while time.time() - t0 < timeout:
        broadcast_hello()
        try:
            data, addr = sock.recvfrom(256)
            if data.startswith(b"ST,"):
                return (addr[0], ESP32_PORT)
        except (socket.timeout, OSError):
            pass
    return None


if ESP32_IP:
    esp32_addr = (ESP32_IP, ESP32_PORT)
else:
    esp32_addr = discover_esp32(DISCOVER_TIMEOUT)
    if esp32_addr is None:
        print("ESP32 not found. Check that the ESP32 is powered on and on the same WiFi, or set ESP32_IP manually.")
        cap.release()
        sock.close()
        exit()
    print("ESP32 found at:", esp32_addr[0])

sock.setblocking(False)  # from here on, do not block while receiving replies
last_esp_rx = time.time()
last_hello = 0.0

# ==============================
# 4. STATE VARIABLES
# ==============================

target_count = 0
no_target_count = 0

TARGET_REQUIRED = 5
NO_TARGET_REQUIRED = 10

target_detected = False
start_time = time.time()
last_timestamp_ms = -1

# Values sent to the ESP32 (keep the old values when the current frame loses the target)
last_err = 0.0   # -1 (offset left) .. +1 (offset right)
last_area = 0.0  # target area / frame area ratio (0..1)

# ==============================
# 5. MAIN PROGRAM
# ==============================

with (
    vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
):
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Lost camera connection!")
            break

        # Flip the image if needed
        # (If you enable flipping, you must invert the sign of last_err below)
        # frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Ensure timestamp_ms is strictly increasing for VIDEO mode
        timestamp_ms = int((time.time() - start_time) * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        # Run detection
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # ==============================
        # 6. CHECK FOR HANDS
        # ==============================

        hand_detected = (
            hand_results.hand_landmarks is not None
            and len(hand_results.hand_landmarks) > 0
        )

        if hand_detected:
            for hand_landmarks in hand_results.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    vision.drawing_styles.get_default_hand_landmarks_style(),
                    vision.drawing_styles.get_default_hand_connections_style(),
                )

        # ==============================
        # 7. CHECK FOR BODY
        # ==============================

        body_detected = False
        pose_lms = None

        if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
            # Take the first detected person
            landmarks = pose_results.pose_landmarks[0]

            visible_points = 0
            for idx in IMPORTANT_POSE_INDICES:
                landmark = landmarks[idx]
                vis = landmark.visibility if landmark.visibility is not None else 0.0
                if vis >= 0.5:
                    visible_points += 1

            # Need at least 2 visible upper-body points
            if visible_points >= 2:
                body_detected = True
                pose_lms = landmarks

            draw_pose_safely(frame, landmarks)

        # ==============================
        # 8. DETERMINE TARGET + COMPUTE err, area
        # ==============================

        current_detection = hand_detected or body_detected

        if current_detection:
            target_count += 1
            no_target_count = 0
        else:
            no_target_count += 1
            target_count = 0

        # Confirm the target over multiple frames
        if target_count >= TARGET_REQUIRED:
            target_detected = True

        if no_target_count >= NO_TARGET_REQUIRED:
            target_detected = False

        # Compute center offset and area from the target's bounding box
        box = None
        if body_detected:
            box = bbox_from_landmarks(pose_lms, min_vis=0.5)
        elif hand_detected:
            box = bbox_from_landmarks(hand_results.hand_landmarks[0])

        if box is not None:
            x0, y0, x1, y1 = box
            cx = (x0 + x1) / 2.0
            last_err = (cx - 0.5) * 2.0          # -1 (left) .. +1 (right)
            last_area = (x1 - x0) * (y1 - y0)    # 0..1

        # ==============================
        # 9. SEND DATA TO ESP32 (UDP)
        # ==============================
        # Format: "D,<det>,<err>,<area>"  (matches pollUdp in the .ino file)

        det = 1 if target_detected else 0
        msg = f"D,{det},{last_err:.3f},{last_area:.3f}"
        try:
            sock.sendto(msg.encode(), esp32_addr)
        except OSError as e:
            print("UDP send error:", e)

        # Read all status replies from the ESP32 (if any) for monitoring
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except (BlockingIOError, OSError):
                break
            print("ESP32:", data.decode(errors="ignore"))
            if data.startswith(b"ST,"):
                esp32_addr = (addr[0], ESP32_PORT)  # update if the ESP32 changes IP
                last_esp_rx = time.time()

        # No reply for > 5 seconds (ESP32 restarted / changed IP) -> search again
        if ESP32_IP is None and time.time() - last_esp_rx > 5 and time.time() - last_hello > 2:
            broadcast_hello()
            last_hello = time.time()

        # ==============================
        # 10. DISPLAY STATUS
        # ==============================

        if target_detected:
            status = "TARGET DETECTED"
            color = (0, 255, 0)
        else:
            status = "NO TARGET"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"err={last_err:+.2f} area={last_area:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("ROBOT HAND TRACKING", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):      # quit
            break
        if key == ord("r"):      # send reset command to the ESP32
            sock.sendto(b"R", esp32_addr)
            print("RESET command sent")

# ==============================
# 11. RELEASE RESOURCES
# ==============================

cap.release()
sock.close()
cv2.destroyAllWindows()

print("Program closed!")