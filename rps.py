import cv2
import numpy as np

def detect_gesture(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return "No hand detected"
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

# Global flags
trigger_next_round = False
exit_game = False

# Define button rectangles
btn_next_round = ((850, 900), (1070, 980))  # top-left, bottom-right
btn_exit = ((1150, 900), (1350, 980))

def mouse_click(event, x, y, flags, param):
    global trigger_next_round, exit_game
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_next_round[0][0] <= x <= btn_next_round[1][0] and btn_next_round[0][1] <= y <= btn_next_round[1][1]:
            trigger_next_round = True
        elif btn_exit[0][0] <= x <= btn_exit[1][0] and btn_exit[0][1] <= y <= btn_exit[1][1]:
            exit_game = True

# Start video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Rock Paper Scissors - Two Player")
cv2.setMouseCallback("Rock Paper Scissors - Two Player", mouse_click)

# Initialize scores
p1_score = p2_score = draws = 0
p1_move = p2_move = result = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1920, 1080))

    # Player 1 ROI
    p1_roi = frame[200:600, 100:600]
    cv2.rectangle(frame, (100, 200), (600, 600), (0, 255, 0), 4)
    cv2.putText(frame, "Player 1", (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Player 2 ROI
    p2_roi = frame[200:600, 1320:1820]
    cv2.rectangle(frame, (1320, 200), (1820, 600), (255, 0, 0), 4)
    cv2.putText(frame, "Player 2", (1320, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    # Display last moves and result
    cv2.putText(frame, f"P1: {p1_move}", (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"P2: {p2_move}", (1320, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    result_text = f"Result: {result}" if result else "Press Next Round to start"
    cv2.putText(frame, result_text, (700, 750), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # Draw score
    score_text = f"Score - P1: {p1_score}  P2: {p2_score}  Draws: {draws}"
    cv2.putText(frame, score_text, (50, 850), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # Draw Buttons
    cv2.rectangle(frame, btn_next_round[0], btn_next_round[1], (0, 255, 255), -1)
    cv2.putText(frame, "Next Round", (btn_next_round[0][0] + 10, btn_next_round[1][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    cv2.rectangle(frame, btn_exit[0], btn_exit[1], (0, 0, 255), -1)
    cv2.putText(frame, "Close", (btn_exit[0][0] + 30, btn_exit[1][1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    # Show Frame
    cv2.imshow("Rock Paper Scissors - Two Player", frame)

    # Wait for round trigger or exit
    if trigger_next_round:
        p1_move = detect_gesture(p1_roi)
        p2_move = detect_gesture(p2_roi)
        result = get_winner(p1_move, p2_move)
        if result == "Player 1 Wins":
            p1_score += 1
        elif result == "Player 2 Wins":
            p2_score += 1
        elif result == "Draw":
            draws += 1
        trigger_next_round = False

    if exit_game or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
