import cv2
import numpy as np
import random
import time
import mediapipe as mp
import os

# --- Setup MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# --- Load Gesture Images ---
gesture_names = ['rock', 'paper', 'scissors', 'gun']
gesture_images = {}
for name in gesture_names:
    path = os.path.join('gestures', f'{name}.png')
    img = cv2.imread(path)
    if img is not None:
        gesture_images[name] = cv2.resize(img, (150, 150))
    else:
        gesture_images[name] = np.zeros((150, 150, 3), dtype=np.uint8)

# --- Gesture Classifier ---
def classify_gesture(hand_landmarks):
    finger_tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:  # Thumb
        fingers.append(1)
    else:
        fingers.append(0)

    for tip in finger_tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)

    if total == 0:
        return 'rock'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'scissors'
    elif total >= 4:
        return 'paper'
    elif fingers == [1, 1, 0, 0, 0]:
        return 'gun'
    else:
        return 'unknown'

# --- Score and Timing ---
score_user, score_computer = 0, 0
last_switch = time.time()
computer_choice = random.choice(gesture_names)

# --- Camera Setup ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    original = frame.copy()

    # --- Threshold Processing ---
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    flood_filled = thresh.copy()
    h, w = thresh.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood_filled, mask, (10, 10), 255)
    flood_filled_inv = cv2.bitwise_not(flood_filled)
    final_mask = cv2.bitwise_or(thresh, flood_filled_inv)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # --- Gesture Detection ---
    rgb_frame = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    user_gesture = 'none'

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(original, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            user_gesture = classify_gesture(hand_landmarks)

    # --- Random Computer Gesture ---
    if time.time() - last_switch > 3:
        computer_choice = random.choice(gesture_names)
        last_switch = time.time()

        # Compare and update score
        if user_gesture != 'none' and user_gesture != 'unknown':
            if user_gesture == computer_choice:
                result = "Tie"
            elif (
                (user_gesture == 'rock' and computer_choice in ['scissors', 'gun']) or
                (user_gesture == 'scissors' and computer_choice == 'paper') or
                (user_gesture == 'paper' and computer_choice == 'rock') or
                (user_gesture == 'gun' and computer_choice != 'gun')
            ):
                result = "You Win"
                score_user += 1
            else:
                result = "You Lose"
                score_computer += 1
        else:
            result = "No Hand"
    else:
        result = "Waiting..."

    # --- Compose Display ---
    cam_small = cv2.resize(original, (320, 240))
    thresh_small = cv2.cvtColor(cv2.resize(final_mask, (320, 240)), cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Place cam & threshold
    canvas[0:240, 0:320] = cam_small
    canvas[0:240, 320:640] = thresh_small

    # Place gesture images
    canvas[260:410, 100:250] = gesture_images.get(user_gesture, np.zeros((150, 150, 3)))
    canvas[260:410, 390:540] = gesture_images[computer_choice]

    # Labels
    cv2.putText(canvas, f'You: {user_gesture}', (90, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f'Computer: {computer_choice}', (360, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f'Result: {result}', (200, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(canvas, f'Score: You {score_user} - {score_computer} Computer', (130, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # Show game window
    cv2.imshow("Rock Paper Scissors Gun", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
