import cv2
import pyautogui
import mediapipe as mp
import os
import urllib.request

# वेबकैम शुरू करें
cap = cv2.VideoCapture(0)

# स्क्रीन का साइज जानने के लिए
screen_width, screen_height = pyautogui.size()
index_y = 0

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'hand_landmarker.task'

# अगर फ़ाइल फोल्डर में नहीं है, तो कोड खुद उसे इंटरनेट से डाउनलोड कर लेगा
if not os.path.exists(model_path):
    print("⏳ मॉडल फ़ाइल आपके कंप्यूटर में नहीं मिली। इंटरनेट से ऑटोमैटिक डाउनलोड हो रही है, कृपया थोड़ा इंतज़ार करें...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    try:
        urllib.request.urlretrieve(url, model_path)
        print("✅ डाउनलोड पूरा हुआ!")
    except Exception as e:
        print(f"❌ डाउनलोड फेल हो गया। कृपया इंटरनेट चेक करें। एरर: {e}")
        cap.release()
        exit()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

print("🚀 वर्चुअल माउस सफलतापूर्वक शुरू हो गया है! बंद करने के लिए विंडो पर 'q' दबाएं।")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = cap.read()
        if not success:
            print("कैमरा चालू नहीं हो पा रहा है।")
            break
            
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # BGR से RGB में बदलें
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # हाथ के पॉइंट्स डिटेक्ट करें
        detection_result = landmarker.detect(mp_image)
        
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0]
            
            # तर्जनी उंगली (Index Finger) - Point 8
            index_finger = hand_landmarks[8]
            x8 = int(index_finger.x * frame_width)
            y8 = int(index_finger.y * frame_height)
            cv2.circle(frame, (x8, y8), 10, (0, 255, 255), -1)
            
            # माउस मूव करें
            mouse_x = screen_width / frame_width * x8
            mouse_y = screen_height / frame_height * y8
            pyautogui.moveTo(mouse_x, mouse_y)
            index_y = mouse_y
            
            # अंगूठा (Thumb) - Point 4
            thumb = hand_landmarks[4]
            x4 = int(thumb.x * frame_width)
            y4 = int(thumb.y * frame_height)
            cv2.circle(frame, (x4, y4), 10, (0, 0, 255), -1)
            
            thumb_y = screen_height / frame_height * y4
            
            # क्लिक लॉजिक
            if abs(index_y - thumb_y) < 25:
                pyautogui.click()
                pyautogui.sleep(0.5)
                            
        cv2.imshow('Virtual Mouse Window', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()