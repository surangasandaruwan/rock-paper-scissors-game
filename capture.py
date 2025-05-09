import cv2

def capture_gesture(filename="gesture.jpg"):
    cap = cv2.VideoCapture(0)
    print("Capturing image in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        cv2.waitKey(1000)  # wait 1 second

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
    cap.release()
    return filename
