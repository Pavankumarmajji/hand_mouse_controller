# gesture_controller.py
import cv2
import time 
import mediapipe as mp
import pyautogui
import numpy as np
from hand_detector import HandDetector

# Screen dimensions
screen_width, screen_height = pyautogui.size()

class GestureController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(max_hands=1)
        
        # Ultra-responsive cursor movement
        self.last_finger_pos = None
        self.velocity_x, self.velocity_y = 0, 0
        self.cursor_speed_factor = 2.0  # Adjust cursor speed (1.0-3.0)
        self.movement_threshold = 2  # Minimum pixel movement to trigger cursor
        
        # Frame settings
        self.frame_reduction = 80  # Reduced for more sensitive movement
        self.cam_width, self.cam_height = 640, 480
        self.cap.set(3, self.cam_width)
        self.cap.set(4, self.cam_height)
        
        # Scrolling
        self.scroll_speed = 15
        self.last_scroll_time = 0
        self.scroll_delay = 0.05  # Faster scrolling
        self.scroll_direction = None
        
        # Click detection
        self.click_threshold = 35
        self.last_click_time = 0
        self.click_delay = 0.3
        
        # Gesture duration for minimize/maximize
        self.gesture_start_time = 0
        self.gesture_hold_duration = 1.2  # 1.2 seconds hold
        self.min_gesture_duration = 0.5   # Minimum 0.5s to start showing progress
        self.is_holding_gesture = False
        self.current_hold_gesture = None
        
        # Recent gestures memory
        self.gesture_history = []
        self.max_history = 5
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def calculate_cursor_position(self, finger_x, finger_y, img_width, img_height):
        """Convert finger position to exact screen position with speed matching"""
        # Map camera coordinates to screen coordinates
        screen_x = np.interp(finger_x, 
                           [self.frame_reduction, img_width - self.frame_reduction],
                           [0, screen_width])
        screen_y = np.interp(finger_y,
                           [self.frame_reduction, img_height - self.frame_reduction],
                           [0, screen_height])
        
        # Calculate movement velocity for instant response
        if self.last_finger_pos:
            delta_x = (finger_x - self.last_finger_pos[0]) * self.cursor_speed_factor
            delta_y = (finger_y - self.last_finger_pos[1]) * self.cursor_speed_factor
            
            # Apply velocity directly (no smoothing for instant response)
            screen_x += delta_x
            screen_y += delta_y
        
        # Store current position for next frame
        self.last_finger_pos = (finger_x, finger_y)
        
        # Boundary checking
        screen_x = np.clip(screen_x, 0, screen_width - 1)
        screen_y = np.clip(screen_y, 0, screen_height - 1)
        
        return int(screen_x), int(screen_y)
    
    def execute_gesture(self, fingers, lm_list, img):
        """Execute actions based on hand gestures"""
        current_time = time.time()
        h, w, _ = img.shape
        
        # 1. CURSOR MOVE: Index finger only - ULTRA RESPONSIVE
        if fingers == [0, 1, 0, 0, 0]:
            self.is_holding_gesture = False
            if len(lm_list) > 8:
                x1, y1 = lm_list[8][1], lm_list[8][2]
                
                # Visual feedback
                cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 255, 255), 2)
                
                # Calculate and move cursor instantly
                cursor_x, cursor_y = self.calculate_cursor_position(x1, y1, w, h)
                pyautogui.moveTo(cursor_x, cursor_y, _pause=False)
                
                # Display movement trail
                if hasattr(self, 'last_cursor_pos'):
                    cv2.line(img, (x1, y1), self.last_cursor_pos, (0, 255, 255), 2)
                self.last_cursor_pos = (x1, y1)
        
        # 2. SCROLL: Index + Middle fingers
        elif fingers == [0, 1, 1, 0, 0]:
            self.is_holding_gesture = False
            if len(lm_list) > 12 and current_time - self.last_scroll_time > self.scroll_delay:
                # Get finger positions
                x1, y1 = lm_list[8][1], lm_list[8][2]  # Index finger
                x2, y2 = lm_list[12][1], lm_list[12][2]  # Middle finger
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Determine scroll direction based on finger orientation
                dx = x2 - x1
                dy = y2 - y1
                
                # Horizontal scroll (left-right)
                if abs(dx) > abs(dy):
                    scroll_x = int(np.interp(dx, [-100, 100], [-self.scroll_speed, self.scroll_speed]))
                    pyautogui.hscroll(scroll_x)
                    direction = "LEFT" if scroll_x < 0 else "RIGHT"
                # Vertical scroll (up-down)
                else:
                    scroll_y = int(np.interp(dy, [-100, 100], [self.scroll_speed, -self.scroll_speed]))
                    pyautogui.scroll(scroll_y)
                    direction = "UP" if scroll_y > 0 else "DOWN"
                
                self.last_scroll_time = current_time
                
                # Visual feedback
                cv2.putText(img, f"SCROLL {direction}", (center_x - 50, center_y - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(img, (center_x, center_y), 8, (0, 255, 0), cv2.FILLED)
        
        # 3. LEFT CLICK: Index finger tap (while both index and middle are open)
        elif fingers == [0, 1, 1, 0, 0] and current_time - self.last_click_time > self.click_delay:
            if len(lm_list) > 8 and len(lm_list) > 12:
                # Check if index finger is significantly lower than middle finger (tap gesture)
                index_y = lm_list[8][2]
                middle_y = lm_list[12][2]
                
                if index_y > middle_y + 20:  # Index finger is lower (tap position)
                    distance = np.linalg.norm(
                        np.array(lm_list[8][1:]) - np.array(lm_list[12][1:])
                    )
                    if distance < self.click_threshold:
                        pyautogui.click()
                        self.last_click_time = current_time
                        cv2.putText(img, "LEFT CLICK", (50, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 4. RIGHT CLICK: Middle finger tap (while both index and middle are open)
        elif fingers == [0, 1, 1, 0, 0] and current_time - self.last_click_time > self.click_delay:
            if len(lm_list) > 8 and len(lm_list) > 12:
                # Check if middle finger is significantly lower than index finger
                index_y = lm_list[8][2]
                middle_y = lm_list[12][2]
                
                if middle_y > index_y + 20:  # Middle finger is lower (right click gesture)
                    distance = np.linalg.norm(
                        np.array(lm_list[8][1:]) - np.array(lm_list[12][1:])
                    )
                    if distance < self.click_threshold:
                        pyautogui.rightClick()
                        self.last_click_time = current_time
                        cv2.putText(img, "RIGHT CLICK", (50, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 5. MINIMIZE ALL TABS: All fingers closed with DURATION
        elif fingers == [0, 0, 0, 0, 0]:
            if self.current_hold_gesture != "minimize":
                self.current_hold_gesture = "minimize"
                self.gesture_start_time = current_time
                self.is_holding_gesture = True
            
            hold_duration = current_time - self.gesture_start_time
            
            # Show progress after minimum duration
            if hold_duration >= self.min_gesture_duration:
                progress = min(hold_duration / self.gesture_hold_duration, 1.0)
                
                # Draw progress bar
                bar_x, bar_y = 50, 200
                bar_width = int(200 * progress)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), 
                            (0, int(255 * progress), 0), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + 200, bar_y + 20), 
                            (255, 255, 255), 2)
                cv2.putText(img, f"MINIMIZE: {int(progress * 100)}%", 
                           (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 165, 255), 2)
                
                # Execute when hold duration reached
                if hold_duration >= self.gesture_hold_duration:
                    pyautogui.hotkey('win', 'd')
                    cv2.putText(img, "ALL TABS MINIMIZED!", (50, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.current_hold_gesture = None
                    self.is_holding_gesture = False
        
        # 6. MAXIMIZE ALL TABS: All fingers open with DURATION
        elif fingers == [1, 1, 1, 1, 1]:
            if self.current_hold_gesture != "maximize":
                self.current_hold_gesture = "maximize"
                self.gesture_start_time = current_time
                self.is_holding_gesture = True
            
            hold_duration = current_time - self.gesture_start_time
            
            # Show progress after minimum duration
            if hold_duration >= self.min_gesture_duration:
                progress = min(hold_duration / self.gesture_hold_duration, 1.0)
                
                # Draw progress bar
                bar_x, bar_y = 50, 300
                bar_width = int(200 * progress)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + 20), 
                            (0, int(255 * progress), 0), cv2.FILLED)
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + 200, bar_y + 20), 
                            (255, 255, 255), 2)
                cv2.putText(img, f"MAXIMIZE: {int(progress * 100)}%", 
                           (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 165, 255), 2)
                
                # Execute when hold duration reached
                if hold_duration >= self.gesture_hold_duration:
                    pyautogui.hotkey('win', 'shift', 'm')
                    cv2.putText(img, "ALL TABS MAXIMIZED!", (50, 350),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.current_hold_gesture = None
                    self.is_holding_gesture = False
        
        # 7. SHIFT TAB: Three fingers (index, middle, ring) - right-left
        elif fingers == [0, 1, 1, 1, 0]:
            self.is_holding_gesture = False
            if len(lm_list) > 16:  # Need up to ring finger tip
                # Get horizontal position of three fingers
                x_positions = [lm_list[8][1], lm_list[12][1], lm_list[16][1]]
                x_avg = sum(x_positions) / 3
                
                # Determine swipe direction
                if hasattr(self, 'last_three_finger_x'):
                    if x_avg > self.last_three_finger_x + 30:  # Swipe right
                        pyautogui.hotkey('shift', 'tab')
                        cv2.putText(img, "SHIFT+TAB (PREVIOUS)", (50, 400),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    elif x_avg < self.last_three_finger_x - 30:  # Swipe left
                        pyautogui.hotkey('ctrl', 'tab')
                        cv2.putText(img, "CTRL+TAB (NEXT)", (50, 400),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
                self.last_three_finger_x = x_avg
                
                # Visual feedback
                for x, y in [(lm_list[8][1], lm_list[8][2]),
                           (lm_list[12][1], lm_list[12][2]),
                           (lm_list[16][1], lm_list[16][2])]:
                    cv2.circle(img, (int(x), int(y)), 8, (255, 255, 0), cv2.FILLED)
        
        # 8. SHIFT APPLICATIONS: Four fingers - right-left
        elif fingers == [0, 1, 1, 1, 1]:
            self.is_holding_gesture = False
            if len(lm_list) > 20:  # Need up to little finger tip
                # Get horizontal position of four fingers
                x_positions = [lm_list[8][1], lm_list[12][1], lm_list[16][1], lm_list[20][1]]
                x_avg = sum(x_positions) / 4
                
                # Determine swipe direction
                if hasattr(self, 'last_four_finger_x'):
                    if x_avg > self.last_four_finger_x + 30:  # Swipe right
                        pyautogui.hotkey('alt', 'shift', 'tab')
                        cv2.putText(img, "ALT+SHIFT+TAB (PREV APP)", (50, 450),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    elif x_avg < self.last_four_finger_x - 30:  # Swipe left
                        pyautogui.hotkey('alt', 'tab')
                        cv2.putText(img, "ALT+TAB (NEXT APP)", (50, 450),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                self.last_four_finger_x = x_avg
                
                # Visual feedback
                for x, y in [(lm_list[8][1], lm_list[8][2]),
                           (lm_list[12][1], lm_list[12][2]),
                           (lm_list[16][1], lm_list[16][2]),
                           (lm_list[20][1], lm_list[20][2])]:
                    cv2.circle(img, (int(x), int(y)), 8, (255, 0, 255), cv2.FILLED)
        
        # 9. RECENT TABS OPEN/CLOSE: Four fingers bottom-up-down
        elif fingers == [0, 1, 1, 1, 1]:
            if len(lm_list) > 20:
                # Get vertical position of four fingers
                y_positions = [lm_list[8][2], lm_list[12][2], lm_list[16][2], lm_list[20][2]]
                y_avg = sum(y_positions) / 4
                
                # Determine vertical swipe direction
                if hasattr(self, 'last_four_finger_y'):
                    if y_avg < self.last_four_finger_y - 40:  # Swipe up (bottom to top)
                        pyautogui.hotkey('ctrl', 'shift', 't')  # Reopen closed tab
                        cv2.putText(img, "REOPEN TAB (Ctrl+Shift+T)", (50, 500),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif y_avg > self.last_four_finger_y + 40:  # Swipe down (top to bottom)
                        pyautogui.hotkey('ctrl', 'w')  # Close current tab
                        cv2.putText(img, "CLOSE TAB (Ctrl+W)", (50, 500),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                self.last_four_finger_y = y_avg
        
        # Reset holding gesture if hand changes
        if not self.is_holding_gesture:
            self.current_hold_gesture = None
        
        # Store gesture in history
        gesture_name = self.get_gesture_name(fingers)
        if gesture_name and (not self.gesture_history or self.gesture_history[-1] != gesture_name):
            self.gesture_history.append(gesture_name)
            if len(self.gesture_history) > self.max_history:
                self.gesture_history.pop(0)
    
    def get_gesture_name(self, fingers):
        """Convert finger array to gesture name"""
        if fingers == [0, 1, 0, 0, 0]:
            return "CURSOR"
        elif fingers == [0, 1, 1, 0, 0]:
            return "SCROLL"
        elif fingers == [0, 0, 0, 0, 0]:
            return "MINIMIZE"
        elif fingers == [1, 1, 1, 1, 1]:
            return "MAXIMIZE"
        elif fingers == [0, 1, 1, 1, 0]:
            return "SHIFT_TAB"
        elif fingers == [0, 1, 1, 1, 1]:
            return "FOUR_FINGER"
        return None
    
    def run(self):
        print("=" * 60)
        print("ULTRA-RESPONSIVE GESTURE CONTROLLER")
        print("=" * 60)
        print("\nGESTURE MAPPINGS:")
        print("1. 👆 Index Finger           : Move Cursor (Ultra-fast)")
        print("2. ✌️ Index+Middle          : Scroll Up/Down/Left/Right")
        print("3. 👇 Index Tap              : Left Click")
        print("4. 👇 Middle Tap             : Right Click")
        print("5. ✊ Fist (Hold 1.2s)       : Minimize All Tabs")
        print("6. ✋ Open Hand (Hold 1.2s)  : Maximize All Tabs")
        print("7. 🤟 Three Fingers Swipe   : Shift+Tab / Ctrl+Tab")
        print("8. 🖐️ Four Fingers Swipe   : Switch Applications")
        print("9. 🖐️ Four Fingers Up/Down : Recent Tabs (Open/Close)")
        print("\nQUICK CONTROLS:")
        print("  '+'  : Increase cursor speed")
        print("  '-'  : Decrease cursor speed")
        print("  'q'  : Quit application")
        print("=" * 60)
        
        while True:
            # Calculate FPS
            current_time = time.time()
            self.frame_count += 1
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            
            success, img = self.cap.read()
            if not success:
                print("Failed to capture video feed")
                break
            
            img = cv2.flip(img, 1)
            img = self.detector.find_hands(img)
            lm_list = self.detector.find_position(img, draw=False)
            
            if lm_list:
                fingers = self.detector.fingers_up(lm_list)
                self.execute_gesture(fingers, lm_list, img)
            
            # Display FPS and status
            cv2.putText(img, f"FPS: {self.fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Speed: {self.cursor_speed_factor:.1f}x", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, f"Hold: {self.gesture_hold_duration:.1f}s", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display gesture history
            y_offset = 120
            for gesture in reversed(self.gesture_history[-3:]):
                cv2.putText(img, f"Last: {gesture}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
            
            # Draw interaction area
            h, w, _ = img.shape
            cv2.rectangle(img, (self.frame_reduction, self.frame_reduction),
                         (w - self.frame_reduction, h - self.frame_reduction),
                         (255, 0, 255), 2)
            
            # Display instructions
            instructions = [
                "CONTROLS:",
                "1 Finger   : Cursor Move (Fast)",
                "2 Fingers  : Scroll",
                "Index Tap  : Left Click",
                "Middle Tap : Right Click",
                "Fist Hold  : Minimize All",
                "Open Hold  : Maximize All",
                "3 Fingers  : Shift Tab",
                "4 Fingers  : Switch Apps",
                "4 Up/Down  : Recent Tabs"
            ]
            
            y_offset = 50
            for instruction in instructions:
                cv2.putText(img, instruction, (w - 250, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            cv2.imshow("Gesture Control v2.0", img)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                self.cursor_speed_factor = min(3.0, self.cursor_speed_factor + 0.1)
                print(f"Cursor speed: {self.cursor_speed_factor:.1f}x")
            elif key == ord('-') or key == ord('_'):
                self.cursor_speed_factor = max(0.5, self.cursor_speed_factor - 0.1)
                print(f"Cursor speed: {self.cursor_speed_factor:.1f}x")
            elif key == ord('['):
                self.gesture_hold_duration = max(0.5, self.gesture_hold_duration - 0.1)
                print(f"Hold duration: {self.gesture_hold_duration:.1f}s")
            elif key == ord(']'):
                self.gesture_hold_duration = min(2.5, self.gesture_hold_duration + 0.1)
                print(f"Hold duration: {self.gesture_hold_duration:.1f}s")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nGesture Controller stopped. Thank you!")

if __name__ == "__main__":
    controller = GestureController()
    controller.run()
