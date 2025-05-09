import cv2
import mediapipe as mp

def recognize_gesture(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True)
    image = cv2.imread(image_path)
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Dummy placeholder logic (extend it with actual finger-count detection logic)
    if results.multi_hand_landmarks:
        # Example: if detected, return something
        return "Rock"  # Replace with actual finger logic later
    return "Unknown"
