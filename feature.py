def greet_player(name):
    return f"welcome to Rock-paper-Scissors, {name}!"

if __name__ == "__main__":
    player_name = input("Enter your name: ")
    print(greet_player(player_name))