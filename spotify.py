import cv2
import mediapipe as mp
import numpy as np
import webbrowser
import time
import platform
import subprocess
from collections import deque
import math

class EmotionMusicPlayer:
    def __init__(self):
        # MediaPipe initialization
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Face mesh for emotion detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Hands for gesture control
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Emotion tracking with stability
        self.emotion_history = deque(maxlen=30)
        self.current_emotion = "neutral"
        self.last_spotify_open_time = 0
        self.spotify_cooldown = 15.0  # 15 seconds cooldown between Spotify opens
        self.emotion_stable_frames = 0
        self.emotion_stability_threshold = 20  # Frames needed for stable emotion
        
        # Enhanced gesture tracking
        self.last_gesture = None
        self.gesture_stable_frames = 0
        self.gesture_threshold = 8
        self.last_gesture_command_time = 0
        self.gesture_cooldown = 1.5  # Reduced for better responsiveness
        self.current_volume = 50  # Track current volume percentage
        
        # Enhanced music configuration with real Spotify playlist IDs
        self.music_config = {
            "happy": {
                "playlists": ["37i9dQZF1DXdPec7aLTmlC", "37i9dQZF1DX0XUsuxWHRQd", "37i9dQZF1DXcBWIGoYBM5M"],
                "description": "Happy & Energetic",
                "search_query": "happy upbeat energetic dance pop"
            },
            "sad": {
                "playlists": ["37i9dQZF1DWX83CujKHHOn", "37i9dQZF1DXdbXrPNafg9d", "37i9dQZF1DWVrtsSlLKzro"],
                "description": "Sad & Melancholic",
                "search_query": "sad melancholy emotional chill lofi"
            },
            "angry": {
                "playlists": ["37i9dQZF1DWWOaP4H0w5b0", "37i9dQZF1DX9qNs32fujYe", "37i9dQZF1DXcF6B6QPhFDv"],
                "description": "Intense & Powerful",
                "search_query": "angry intense rock metal powerful"
            },
            "surprised": {
                "playlists": ["37i9dQZF1DX0BcQWzuB7ZO", "37i9dQZF1DXcBWIGoYBM5M", "37i9dQZF1DX4JAvHpjipBk"],
                "description": "Exciting & Dynamic",
                "search_query": "surprising exciting dynamic upbeat electronic"
            },
            "neutral": {
                "playlists": ["37i9dQZF1DX0XUsuxWHRQd", "37i9dQZF1DWVrtsSlLKzro", "37i9dQZF1DX4JAvHpjipBk"],
                "description": "Chill & Ambient",
                "search_query": "chill ambient focus background study"
            }
        }
        
        print("🎵 Enhanced Emotion + Gesture Music Player v2.0")
        print("=" * 55)
        print("🎭 Emotion Detection:")
        print("   - Detects: Happy, Sad, Angry, Surprised, Neutral")
        print("   - Auto-opens Spotify playlists (15s cooldown)")
        print("👋 Extended Gesture Controls:")
        print("   - ✊ Fist = Play/Pause Music")
        print("   - ✋ Open Palm = Stop/Pause Music")
        print("   - ✌️ Peace Sign = Next Song")
        print("   - 👍 Thumb Up = Previous Song")
        print("   - ☝️ Point Up = Volume Up")
        print("   - 👇 Point Down = Volume Down")
        print("⚙️  Press 'q' to quit")
        print("=" * 55)
    
    def calculate_emotion(self, landmarks):
        """Enhanced emotion calculation with better accuracy"""
        if not landmarks:
            return "neutral"
        
        h, w = 480, 640
        points = np.array([[landmark.x * w, landmark.y * h] for landmark in landmarks.landmark])
        
        # Key facial landmarks
        left_mouth = points[61]      # Left mouth corner
        right_mouth = points[291]    # Right mouth corner
        top_lip = points[13]         # Upper lip center
        bottom_lip = points[14]      # Lower lip center
        
        # Eye landmarks
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386] 
        right_eye_bottom = points[374]
        
        # Eyebrow landmarks
        left_eyebrow = points[70]
        right_eyebrow = points[300]
        
        # Calculate features
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        mouth_width = abs(right_mouth[0] - left_mouth[0])
        lip_distance = abs(top_lip[1] - bottom_lip[1])
        
        # Eye openness
        left_eye_openness = abs(left_eye_top[1] - left_eye_bottom[1])
        right_eye_openness = abs(right_eye_top[1] - right_eye_bottom[1])
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2
        
        # Eyebrow position
        avg_eyebrow_height = (left_eyebrow[1] + right_eyebrow[1]) / 2
        avg_eye_height = (left_eye_top[1] + right_eye_top[1]) / 2
        
        # Mouth curve calculation
        mouth_curve_ratio = (mouth_center_y - top_lip[1]) / mouth_width if mouth_width > 0 else 0
        
        # Enhanced emotion classification
        if mouth_curve_ratio < -0.015 and avg_eye_openness > 6:
            return "happy"
        elif mouth_curve_ratio > 0.015 and avg_eye_openness < 8:
            return "sad"
        elif avg_eye_openness > 15 and lip_distance > 8:
            return "surprised"
        elif avg_eyebrow_height < avg_eye_height - 8 and lip_distance < 4:
            return "angry"
        else:
            return "neutral"
    
    def detect_gesture(self, hand_landmarks):
        """Enhanced gesture detection with new controls"""
        if not hand_landmarks:
            return None
        
        landmarks = hand_landmarks.landmark
        
        # Fingertip and joint positions
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_joints = [3, 6, 10, 14, 18]  # MCP joints
        finger_pips = [2, 5, 9, 13, 17]   # PIP joints
        
        fingers_up = []
        
        # Thumb (special case due to orientation)
        if landmarks[fingertips[0]].x > landmarks[finger_joints[0]].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # Other fingers
        for i in range(1, 5):
            if landmarks[fingertips[i]].y < landmarks[finger_joints[i]].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        total_fingers = sum(fingers_up)
        
        # Enhanced gesture classification
        if total_fingers == 0:
            return "Fist"
        elif total_fingers == 5:
            return "Open_Palm"
        elif fingers_up == [0, 1, 1, 0, 0]:  # Index and middle up
            return "Peace_Sign"
        elif fingers_up == [1, 0, 0, 0, 0]:  # Only thumb up
            return "Thumb_Up"
        elif fingers_up == [0, 1, 0, 0, 0]:  # Only index finger up
            # Check if pointing up or down based on finger direction
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            if index_tip.y < index_pip.y - 0.05:  # Pointing up
                return "Point_Up"
            elif index_tip.y > index_pip.y + 0.05:  # Pointing down
                return "Point_Down"
            else:
                return "Point_Neutral"
        else:
            return "Unknown"
    
    def send_media_command(self, command):
        """Enhanced media control commands with volume and track controls"""
        try:
            system = platform.system().lower()
            
            if system == "windows":
                try:
                    import win32api
                    import win32con
                    
                    keys = {
                        "play": 0xB3,      # VK_MEDIA_PLAY_PAUSE
                        "pause": 0xB3,     # VK_MEDIA_PLAY_PAUSE
                        "next": 0xB0,      # VK_MEDIA_NEXT_TRACK
                        "previous": 0xB1,   # VK_MEDIA_PREV_TRACK
                        "volume_up": 0xAF,  # VK_VOLUME_UP
                        "volume_down": 0xAE # VK_VOLUME_DOWN
                    }
                    
                    if command in keys:
                        win32api.keybd_event(keys[command], 0, 0, 0)
                        win32api.keybd_event(keys[command], 0, win32con.KEYEVENTF_KEYUP, 0)
                        return True
                        
                except ImportError:
                    print("💡 Install pywin32 for Windows media keys: pip install pywin32")
                    return False
                    
            elif system == "darwin":  # macOS
                applescript_commands = {
                    "play": "tell application \"Music\" to play",
                    "pause": "tell application \"Music\" to pause",
                    "next": "tell application \"Music\" to next track",
                    "previous": "tell application \"Music\" to previous track",
                    "volume_up": "set volume output volume (output volume of (get volume settings) + 10)",
                    "volume_down": "set volume output volume (output volume of (get volume settings) - 10)"
                }
                
                if command in applescript_commands:
                    subprocess.run(["osascript", "-e", applescript_commands[command]], 
                                 capture_output=True, text=True)
                    return True
                    
            elif system == "linux":
                playerctl_commands = {
                    "play": "play",
                    "pause": "pause",
                    "next": "next",
                    "previous": "previous"
                }
                
                if command in playerctl_commands:
                    result = subprocess.run(["playerctl", playerctl_commands[command]], 
                                          capture_output=True, text=True)
                    success = result.returncode == 0
                elif command in ["volume_up", "volume_down"]:
                    # Use amixer for volume control on Linux
                    volume_change = "+5%" if command == "volume_up" else "-5%"
                    result = subprocess.run(["amixer", "set", "Master", volume_change], 
                                          capture_output=True, text=True)
                    success = result.returncode == 0
                else:
                    success = False
                
                return success
                    
        except Exception as e:
            print(f"⚠️ Media command error: {e}")
            return False
        
        return False
    
    def process_gesture_command(self, gesture):
        """Enhanced gesture command processing with new controls"""
        current_time = time.time()
        if current_time - self.last_gesture_command_time < self.gesture_cooldown:
            return False
        
        command_sent = False
        display_message = ""
        
        if gesture == "Fist":
            print("✊ FIST → ▶️ PLAY/PAUSE")
            command_sent = self.send_media_command("play")
            display_message = "PLAY/PAUSE"
            
        elif gesture == "Open_Palm":
            print("✋ OPEN PALM → ⏸️ PAUSE")
            command_sent = self.send_media_command("pause")
            display_message = "PAUSE"
            
        elif gesture == "Peace_Sign":
            print("✌️ PEACE SIGN → ⏭️ NEXT SONG")
            command_sent = self.send_media_command("next")
            display_message = "NEXT SONG"
            
        elif gesture == "Thumb_Up":
            print("👍 THUMB UP → ⏮️ PREVIOUS SONG")
            command_sent = self.send_media_command("previous")
            display_message = "PREVIOUS SONG"
            
        elif gesture == "Point_Up":
            print("☝️ POINT UP → 🔊 VOLUME UP")
            command_sent = self.send_media_command("volume_up")
            self.current_volume = min(100, self.current_volume + 10)
            display_message = f"VOLUME UP ({self.current_volume}%)"
            
        elif gesture == "Point_Down":
            print("👇 POINT DOWN → 🔉 VOLUME DOWN")
            command_sent = self.send_media_command("volume_down")
            self.current_volume = max(0, self.current_volume - 10)
            display_message = f"VOLUME DOWN ({self.current_volume}%)"
        
        if command_sent:
            self.last_gesture_command_time = current_time
            return display_message
        
        return False
    
    def smooth_emotion(self, emotion):
        """Smooth emotion detection with stability checking"""
        self.emotion_history.append(emotion)
        
        # Count occurrences of each emotion
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        # Get most frequent emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        confidence = emotion_counts[dominant_emotion] / len(self.emotion_history)
        
        return dominant_emotion, confidence
    
    def should_open_spotify(self, emotion, confidence):
        """Determine if Spotify should be opened based on emotion stability and cooldown"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_spotify_open_time < self.spotify_cooldown:
            return False
        
        # Check if emotion is different from current
        if emotion == self.current_emotion:
            return False
        
        # Check confidence threshold
        if confidence < 0.7:
            return False
        
        # Don't open for neutral emotion
        if emotion == "neutral":
            return False
        
        return True
    
    def open_spotify_music(self, emotion):
        """Open Spotify with emotion-based playlist"""
        if emotion not in self.music_config:
            return False
        
        config = self.music_config[emotion]
        
        try:
            # Try to open with Spotify app first
            playlist_id = np.random.choice(config["playlists"])
            spotify_url = f"spotify:playlist:{playlist_id}"
            
            print(f"🎧 Opening {config['description']} playlist for {emotion.upper()} mood")
            
            # Try Spotify desktop app first
            try:
                webbrowser.open(spotify_url)
            except:
                # Fallback to web player
                web_url = f"https://open.spotify.com/playlist/{playlist_id}"
                webbrowser.open(web_url)
            
            self.last_spotify_open_time = time.time()
            self.current_emotion = emotion
            return True
            
        except Exception as e:
            print(f"⚠️ Error opening Spotify: {e}")
            return False
    
    def draw_gesture_guide(self, image):
        """Draw gesture guide on the side of the screen"""
        guide_x = image.shape[1] - 250
        guide_y = 30
        
        gestures = [
            ("✊ Fist = Play/Pause", (0, 255, 0)),
            ("✋ Palm = Pause", (0, 255, 0)),
            ("✌️ Peace = Next", (255, 255, 0)),
            ("👍 Thumb = Previous", (255, 255, 0)),
            ("☝️ Up = Vol+", (255, 0, 255)),
            ("👇 Down = Vol-", (255, 0, 255))
        ]
        
        for i, (gesture_text, color) in enumerate(gestures):
            y_pos = guide_y + (i * 25)
            cv2.putText(image, gesture_text, (guide_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_emotion_info(self, image, emotion, confidence, gesture_info=""):
        """Enhanced information display with volume indicator"""
        emotion_colors = {
            "happy": (0, 255, 0),      # Green
            "sad": (255, 0, 0),        # Blue  
            "angry": (0, 0, 255),      # Red
            "surprised": (0, 255, 255), # Yellow
            "neutral": (128, 128, 128)  # Gray
        }
        
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw emotion text
        cv2.putText(image, f"Emotion: {emotion.upper()}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw confidence bar
        bar_width = int(confidence * 200)
        cv2.rectangle(image, (10, 60), (10 + bar_width, 80), color, -1)
        cv2.rectangle(image, (10, 60), (210, 80), (255, 255, 255), 2)
        cv2.putText(image, f"{confidence:.1%}", (220, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw gesture info
        if gesture_info:
            cv2.putText(image, f"Command: {gesture_info}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw volume indicator
        volume_bar_width = int(self.current_volume * 2)  # Scale to 200px max
        cv2.rectangle(image, (10, 130), (10 + volume_bar_width, 150), (255, 255, 0), -1)
        cv2.rectangle(image, (10, 130), (210, 150), (255, 255, 255), 2)
        cv2.putText(image, f"Volume: {self.current_volume}%", (220, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw cooldown info
        current_time = time.time()
        remaining_cooldown = self.spotify_cooldown - (current_time - self.last_spotify_open_time)
        if remaining_cooldown > 0:
            cv2.putText(image, f"Spotify cooldown: {remaining_cooldown:.1f}s", (10, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw gesture guide
        self.draw_gesture_guide(image)
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("📷 Camera started! Show your face and hands...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process face for emotion detection
                face_results = self.face_mesh.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)
                
                current_emotion = "neutral"
                confidence = 0.0
                gesture_info = ""
                
                # Emotion Detection
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # Draw face landmarks (minimal)
                        self.mp_drawing.draw_landmarks(
                            frame, face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(0, 255, 0), thickness=1, circle_radius=1
                            )
                        )
                        
                        # Calculate emotion
                        detected_emotion = self.calculate_emotion(face_landmarks)
                        smoothed_emotion, confidence = self.smooth_emotion(detected_emotion)
                        current_emotion = smoothed_emotion
                        
                        # Check if we should open Spotify
                        if self.should_open_spotify(smoothed_emotion, confidence):
                            success = self.open_spotify_music(smoothed_emotion)
                            if success:
                                print(f"✅ Opened Spotify for {smoothed_emotion} emotion")
                
                # Enhanced Gesture Detection
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Draw hand landmarks
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                        
                        # Detect gesture
                        detected_gesture = self.detect_gesture(hand_landmarks)
                        
                        if detected_gesture == self.last_gesture and detected_gesture in [
                            "Fist", "Open_Palm", "Peace_Sign", "Thumb_Up", "Point_Up", "Point_Down"
                        ]:
                            self.gesture_stable_frames += 1
                        else:
                            self.gesture_stable_frames = 0
                            self.last_gesture = detected_gesture
                        
                        # Execute gesture command if stable
                        if self.gesture_stable_frames >= self.gesture_threshold:
                            command_result = self.process_gesture_command(detected_gesture)
                            if command_result:
                                gesture_info = command_result
                            self.gesture_stable_frames = 0
                        
                        # Show current gesture
                        if detected_gesture and detected_gesture != "Unknown":
                            if not gesture_info:
                                gesture_info = f"Detecting: {detected_gesture}"
                
                # Draw information overlay
                self.draw_emotion_info(frame, current_emotion, confidence, gesture_info)
                
                # Show frame
                cv2.imshow('Enhanced Emotion + Gesture Music Player v2.0 - Press Q to quit', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Stopping...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🎵 Enhanced Music Player stopped!")

if __name__ == "__main__":
    try:
        print("🚀 Starting Enhanced Emotion + Gesture Music Player v2.0...")
        print("📦 Required packages: opencv-python, mediapipe, numpy")
        print("🔧 Optional: pywin32 (Windows), playerctl+amixer (Linux)")
        print()
        
        player = EmotionMusicPlayer()
        player.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure to install: pip install opencv-python mediapipe numpy")
        if platform.system().lower() == "windows":
            print("For Windows media keys: pip install pywin32")
        elif platform.system().lower() == "linux":
            print("For Linux: sudo apt install playerctl alsa-utils")