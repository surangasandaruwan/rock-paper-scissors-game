import cv2
import numpy as np

# Define gesture detection function
def detect_gesture(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Show grayscale and thresholding
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Threshold", thresh)

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
    elif finger_count == 2:
        return "scissors"
    elif finger_count == 5:
        return "paper"
    elif finger_count == 4:
        return "lizard"
    else:
        return "spock"

# Define the function to determine the winner
def get_winner(p1, p2):
    # Adding the new rules for Lizard and Spock
    rules = {
        ("rock", "scissors"): "Player 1 Wins",
        ("scissors", "paper"): "Player 1 Wins",
        ("paper", "rock"): "Player 1 Wins",
        ("rock", "lizard"): "Player 1 Wins",
        ("lizard", "spock"): "Player 1 Wins",
        ("spock", "scissors"): "Player 1 Wins",
        ("scissors", "rock"): "Player 2 Wins",
        ("paper", "scissors"): "Player 2 Wins",
        ("rock", "paper"): "Player 2 Wins",
        ("lizard", "rock"): "Player 2 Wins",
        ("spock", "lizard"): "Player 2 Wins",
        ("scissors", "spock"): "Player 2 Wins",
    }

    if p1 == p2:
        return "Draw"
    else:
        return rules.get((p1, p2), "Invalid Move")
    

cap = cv2.VideoCapture(0)
cv2.namedWindow("Rock Paper Scissors - Two Player")
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at ({x}, {y})")

cv2.setMouseCallback("Rock Paper Scissors - Two Player", mouse_click)

# Initialize scores
p1_score = p2_score = draws = 0
p1_move = p2_move = result = ""

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

    # Player 2 ROI
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

    stop_game = False;

    # Detect moves continuously
    if not stop_game:
        p1_move = detect_gesture(p1_roi)
        p2_move = detect_gesture(p2_roi)
        result = get_winner(p1_move, p2_move)
        if result == "Player 1 Wins":
            p1_score += 1
        elif result == "Player 2 Wins":
            p2_score += 1
        elif result == "Draw":
            draws += 1

    # Check for stop or exit
    if stop_game:
        break
    exit_game = False
    if exit_game or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
