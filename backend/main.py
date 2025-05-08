# Rock Paper Scissors - Two Player Game using Hand Gestures
# This code uses OpenCV and MediaPipe to detect hand gestures for a two-player game of Rock Paper Scissors.

import cv2
import mediapipe as mp

# === Game logic ===
def decide_winner(p1, p2):
    rules = {
        "rock": ["scissors", "lizard"],
        "paper": ["rock", "spock"],
        "scissors": ["paper", "lizard"],
        "lizard": ["spock", "paper"],
        "spock": ["scissors", "rock"]
    }

    if p1 == p2:
        return "Draw"
    elif p2 in rules.get(p1, []):
        return "Player 1 Wins"
    else:
        return "Player 2 Wins"

# === Hand detection setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

def get_gesture(lm, hand_label):
    try:
        tips = [8, 12, 16, 20]
        fingers = []

        # Index to pinky
        for tip in tips:
            if lm[tip].y < lm[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Thumb (based on handedness)
        if hand_label == "Left":
            fingers.insert(0, 1 if lm[4].x > lm[3].x else 0)
        else:
            fingers.insert(0, 1 if lm[4].x < lm[3].x else 0)

        total = fingers.count(1)
        if total == 0:
            return "rock"
        elif total == 2:
            return "scissors"
        elif total == 5:
            return "paper"
        else:
            return "unknown"
    except Exception as e:
        print(f"[ERROR] Gesture detection failed: {e}")
        return "unknown"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("[ERROR] Failed to capture image from webcam.")
        break

    image = cv2.flip(image, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_list = []

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            try:
                hand_label = results.multi_handedness[idx].classification[0].label
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                gesture = get_gesture(lm, hand_label)
                gesture_list.append(gesture)
            except Exception as e:
                print(f"[ERROR] Failed to process hand {idx}: {e}")

    # Display text
    text = ""
    if len(gesture_list) == 2:
        p1, p2 = gesture_list[0], gesture_list[1]
        result = decide_winner(p1, p2)
        text = f"{p1.upper()} vs {p2.upper()} → {result}"
    elif len(gesture_list) == 1:
        text = f"{gesture_list[0].upper()} (waiting for second player)"

    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.imshow('Rock Paper Scissors - Two Player', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# End of code
# This code is a simple implementation of a two-player Rock Paper Scissors game using hand gestures.