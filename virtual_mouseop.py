import cv2
import pyautogui
import mediapipe as mp
import os
import urllib.request

# PyAutoGUI performance boost
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

screen_width, screen_height = pyautogui.size()
model_path = 'hand_landmarker.task'

def download_model():
    # Professional status logging
    print(">>> [SYSTEM] Checking dependencies...")
    print(">>> [SYSTEM] Model file not found. Initializing secure download from Google servers...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print(">>> [SUCCESS] Model file successfully deployed.")

if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    if os.path.exists(model_path):
        os.remove(model_path)
    download_model()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

smoothening = 5
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Clean startup message
print("="*50)
print(">>> VIRTUAL MOUSE INTERFACE [INITIALIZED]")
print(">>> Status: Running at 640x480 resolution")
print(">>> Controls: Index Finger to move | Thumb pinch to click")
print(">>> Press 'q' to terminate the session.")
print("="*50)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            index_finger = hand_landmarks[8]
            x8 = int(index_finger.x * frame_width)
            y8 = int(index_finger.y * frame_height)
            cv2.circle(frame, (x8, y8), 10, (0, 255, 255), -1)
            
            mouse_x = screen_width / frame_width * x8
            mouse_y = screen_height / frame_height * y8
            
            clocX = plocX + (mouse_x - plocX) / smoothening
            clocY = plocY + (mouse_y - plocY) / smoothening
            
            pyautogui.moveTo(clocX, clocY)
            plocX, plocY = clocX, clocY
            
            thumb = hand_landmarks[4]
            x4 = int(thumb.x * frame_width)
            y4 = int(thumb.y * frame_height)
            cv2.circle(frame, (x4, y4), 10, (0, 0, 255), -1)
            
            if abs(y8 - y4) < 30 and abs(x8 - x4) < 30:
                pyautogui.click()
                pyautogui.sleep(0.3)
                            
        cv2.imshow('Pro Virtual Mouse', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(">>> [SYSTEM] Session terminated by user.")
            break

cap.release()
cv2.destroyAllWindows()