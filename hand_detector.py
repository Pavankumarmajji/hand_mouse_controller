import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_conf=0.7, track_conf=0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_conf,
            min_tracking_confidence=self.track_conf
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        if self.results.multi_hand_landmarks and draw:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw and id in [4, 8, 12, 16, 20]:  # Fingertips
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lm_list
    
    def fingers_up(self, lm_list):
        fingers = []
        
        # Thumb
        if lm_list[4][1] > lm_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip_id, pip_id in zip(tip_ids, pip_ids):
            if lm_list[tip_id][2] < lm_list[pip_id][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers