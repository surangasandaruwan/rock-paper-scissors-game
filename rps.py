import cv2
import numpy as np

def detect_gesture(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "No hand detected"  # Return "No hand detected" if no contours found

    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt, returnPoints=False)

    if len(hull) < 3:
        return "rock"

    defects = cv2.convexityDefects(cnt, hull)

    if defects is None:
        return "rock"

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))
        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))

        if angle <= np.pi / 2:
            finger_count += 1

    if finger_count == 0:
        return "rock"
    elif finger_count <= 2:
        return "scissors"
    else:
        return "paper"

def get_winner(p1, p2):
    if p1 == p2:
        return "Draw"
    elif (p1 == "rock" and p2 == "scissors") or \
         (p1 == "scissors" and p2 == "paper") or \
         (p1 == "paper" and p2 == "rock"):
        return "Player 1 Wins"
    else:
        return "Player 2 Wins"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1920, 1080))  # Resize to full HD


    # Define ROI for Player 1 (left side)
    p1_roi = frame[200:600, 100:600]  # [y1:y2, x1:x2]
    cv2.rectangle(frame, (100, 200), (600, 600), (0, 255, 0), 4)
    cv2.putText(frame, "Player 1", (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Define ROI for Player 2 (right side)
    p2_roi = frame[200:600, 1320:1820]
    cv2.rectangle(frame, (1320, 200), (1820, 600), (255, 0, 0), 4)
    cv2.putText(frame, "Player 2", (1320, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)


    # Detect gesture for both players
    p1_move = detect_gesture(p1_roi)
    p2_move = detect_gesture(p2_roi)

    # Show moves for each player
    p1_move_text = f"P1: {p1_move if p1_move != 'No hand detected' else 'No hand detected'}"
    p2_move_text = f"P2: {p2_move if p2_move != 'No hand detected' else 'No hand detected'}"

    cv2.putText(frame, p1_move_text, (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, p2_move_text, (1320, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Determine the result
    result = get_winner(p1_move, p2_move)

    # Show result at the bottom
    text = f"Result: {result}"
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    x_position = (1920 - text_width) // 2
    cv2.putText(frame, text, (x_position, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)


    # Create full-screen window
    cv2.namedWindow("Rock Paper Scissors - Two Player", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rock Paper Scissors - Two Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Display the frame
    cv2.imshow("Rock Paper Scissors - Two Player", frame)

    # Exit loop when 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
