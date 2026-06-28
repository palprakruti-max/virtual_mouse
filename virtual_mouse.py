import cv2
import pyautogui
import mediapipe as mp
import os
import urllib.request

# Screen ka size nikalne ke liye
screen_width, screen_height = pyautogui.size()
index_y = 0

model_path = 'hand_landmarker.task'

# Corrupt file ko delete karke sahi file download karne ka function
def download_model():
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
    print("⏳ Model file download ho rahi hai (Lagbhag 3 MB)... Kripya internet on rakhein.")
    # File ko browser ki tarah download karne ke liye (taaki corrupt na ho)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("✅ Download poora hua! File ekdum sahi hai.")

# Agar file nahi hai, ya corrupt hai (size 1MB se kam hai), toh phir se download karo
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    if os.path.exists(model_path):
        os.remove(model_path) # Purani corrupt file hata do
    download_model()

# Naya Task API (Jo Python 3.14 me chal raha hai)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)
print("virtual mouse started press 'q' to exit")

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
            
            # Tarjani ungli (Index Finger - Point 8)
            index_finger = hand_landmarks[8]
            x8 = int(index_finger.x * frame_width)
            y8 = int(index_finger.y * frame_height)
            cv2.circle(frame, (x8, y8), 10, (0, 255, 255), -1)
            
            mouse_x = screen_width / frame_width * x8
            mouse_y = screen_height / frame_height * y8
            pyautogui.moveTo(mouse_x, mouse_y)
            index_y = mouse_y
            
            # Angutha (Thumb - Point 4)
            thumb = hand_landmarks[4]
            x4 = int(thumb.x * frame_width)
            y4 = int(thumb.y * frame_height)
            cv2.circle(frame, (x4, y4), 10, (0, 0, 255), -1)
            
            thumb_y = screen_height / frame_height * y4
            
            # Click logic (Ungli aur angutha paas aane par)
            if abs(index_y - thumb_y) < 25:
                pyautogui.click()
                pyautogui.sleep(0.5)
                            
        cv2.imshow('Virtual Mouse Window', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()