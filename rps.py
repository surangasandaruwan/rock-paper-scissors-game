import cv2
import numpy as np

# Constants
WINDOW_NAME = "Rock Paper Scissors - Two Player"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
ROI_TOP, ROI_BOTTOM = 200, 600
P1_LEFT, P1_RIGHT = 100, 600
P2_LEFT, P2_RIGHT = 1320, 1820
FONT = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD_ANGLE = np.pi / 2
TEXT_COLOR_RESULT = (0, 0, 255)
TEXT_COLOR_P1 = (0, 255, 0)
TEXT_COLOR_P2 = (255, 0, 0)

def detect_gesture(region_of_interest):
    """Detects hand gesture from the given region of interest (ROI)."""
    gray = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("Grayscale", gray)
    cv2.imshow("Threshold", thresh)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No hand detected"

    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    if len(hull) < 3:
        return "rock"

    defects = cv2.convexityDefects(largest_contour, hull)
    if defects is None:
        return "rock"

    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c + 1e-5))
        if angle <= THRESHOLD_ANGLE:
            finger_count += 1

    return classify_gesture(finger_count)

def classify_gesture(finger_count):
    """Returns gesture label based on detected finger count."""
    gesture_map = {
        0: "rock",
        2: "scissors",
        4: "lizard",
        5: "paper"
    }
    return gesture_map.get(finger_count, "spock")

def get_winner(p1_gesture, p2_gesture):
    """Returns the result of the game based on both players' gestures."""
    rules = {
        ("rock", "scissors"), ("rock", "lizard"),
        ("scissors", "paper"), ("scissors", "lizard"),
        ("paper", "rock"), ("paper", "spock"),
        ("lizard", "spock"), ("lizard", "paper"),
        ("spock", "scissors"), ("spock", "rock")
    }

    if p1_gesture == p2_gesture:
        return "Draw"
    elif (p1_gesture, p2_gesture) in rules:
        return "Player 1 Wins"
    elif (p2_gesture, p1_gesture) in rules:
        return "Player 2 Wins"
    else:
        return "Invalid Move"

def draw_ui(frame, p1_gesture, p2_gesture, result):
    """Draws all UI elements on the frame."""
    cv2.rectangle(frame, (P1_LEFT, ROI_TOP), (P1_RIGHT, ROI_BOTTOM), TEXT_COLOR_P1, 4)
    cv2.putText(frame, "Player 1", (P1_LEFT, ROI_TOP - 10), FONT, 1.2, TEXT_COLOR_P1, 3)

    cv2.rectangle(frame, (P2_LEFT, ROI_TOP), (P2_RIGHT, ROI_BOTTOM), TEXT_COLOR_P2, 4)
    cv2.putText(frame, "Player 2", (P2_LEFT, ROI_TOP - 10), FONT, 1.2, TEXT_COLOR_P2, 3)

    cv2.putText(frame, f"P1: {p1_gesture}", (P1_LEFT, 650), FONT, 1, TEXT_COLOR_P1, 2)
    cv2.putText(frame, f"P2: {p2_gesture}", (P2_LEFT, 650), FONT, 1, TEXT_COLOR_P2, 2)

    result_text = f"Result: {result}"
    (text_width, _), _ = cv2.getTextSize(result_text, FONT, 1.5, 4)
    x_pos = (FRAME_WIDTH - text_width) // 2
    cv2.putText(frame, result_text, (x_pos, 1000), FONT, 1.5, TEXT_COLOR_RESULT, 4)

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    player_scores = {'Player 1': 0, 'Player 2': 0, 'Draws': 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Define ROIs
        p1_roi = frame[ROI_TOP:ROI_BOTTOM, P1_LEFT:P1_RIGHT]
        p2_roi = frame[ROI_TOP:ROI_BOTTOM, P2_LEFT:P2_RIGHT]

        # Detect gestures
        p1_gesture = detect_gesture(p1_roi)
        p2_gesture = detect_gesture(p2_roi)

        # Determine winner
        result = get_winner(p1_gesture, p2_gesture)
        if result == "Player 1 Wins":
            player_scores['Player 1'] += 1
        elif result == "Player 2 Wins":
            player_scores['Player 2'] += 1
        elif result == "Draw":
            player_scores['Draws'] += 1

        # Draw interface
        draw_ui(frame, p1_gesture, p2_gesture, result)
        cv2.imshow(WINDOW_NAME, frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final Scores:", player_scores)

if __name__ == "__main__":
    main()
