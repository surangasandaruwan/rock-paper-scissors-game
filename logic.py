import random

choices = ["Rock", "Paper", "Scissors", "Lizard", "Spock"]

rules = {
    "Rock": ["Scissors", "Lizard"],
    "Paper": ["Rock", "Spock"],
    "Scissors": ["Paper", "Lizard"],
    "Lizard": ["Spock", "Paper"],
    "Spock": ["Scissors", "Rock"]
}

def get_computer_choice():
    return random.choice(choices)

def decide_winner(player, computer):
    if player == computer:
        return "It's a tie!"
    elif computer in rules.get(player, []):
        return "You win!"
    else:
        return "Computer wins!"
