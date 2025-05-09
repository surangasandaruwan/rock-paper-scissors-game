from flask import Flask, render_template, request
from capture import capture_gesture
from gesture_utils import recognize_gesture
from logic import get_computer_choice, decide_winner
import pygame
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/play", methods=["POST"])
def play():
    # Play countdown audio
    pygame.mixer.init()
    pygame.mixer.music.load("sounds/countdown.mp3")
    pygame.mixer.music.play()
    time.sleep(6)  # wait for countdown to finish

    image_path = capture_gesture()
    player = recognize_gesture(image_path)
    computer = get_computer_choice()
    result = decide_winner(player, computer)

    return render_template("index.html", player=player, computer=computer, result=result)

if __name__ == "__main__":
    app.run(debug=True)
