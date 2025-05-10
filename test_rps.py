import unittest
from rps import determine_winner  # Adjust this import if needed

class TestRPS(unittest.TestCase):
    def test_rock_vs_scissors(self):
        self.assertEqual(determine_winner('rock', 'scissors'), 'rock')

    def test_scissors_vs_paper(self):
        self.assertEqual(determine_winner('scissors', 'paper'), 'scissors')

    def test_paper_vs_rock(self):
        self.assertEqual(determine_winner('paper', 'rock'), 'paper')

if __name__ == '__main__':
    unittest.main()
    
