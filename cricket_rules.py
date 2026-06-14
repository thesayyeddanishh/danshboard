# cricket_rules.py

# This dictionary holds all the custom logic for every format.
FORMAT_CONFIG = {
    "Men's T20": {
        "ball_type": "White",
        "seam_bins": {"Full Toss": [-4, 0.5], "Yorker": [0.5, 2.5], "The Slot": [2.5, 5.8], "Length": [5.8, 8], "Short": [8, 10], "Bouncer": [10, 16]},
        "spin_bins": {"OP": [-2, 2.8], "Full": [2.8, 4.4], "Good": [4.4, 6.2], "Short": [6.2, 15.0]}
    },
    "Women's T20": {
        "ball_type": "White",
        "seam_bins": {"Full Toss": [-5, 0.9], "Yorker": [0.9, 2.8], "The Slot": [2.8, 5.5], "Length": [5.5, 8], "Short": [8, 16]},
        "spin_bins": {"OP": [-2, 2.5], "Full": [2.5, 4], "Good": [4, 5.7], "Short": [5.7, 12.0]}
    },
    "Men's RedBall": {
        "ball_type": "Red",
        "seam_bins": {"Full": [-4, 5.8], "Length": [5.8, 8], "Short": [8, 10], "Bouncer": [10, 16]},
        "spin_bins": {"OP": [-2, 2.8], "Full": [2.8, 4.4], "Good": [4.4, 6.2], "Short": [6.2, 15.0]}
    },
    "Men's RedBall - AUS": {
        "ball_type": "Red",
        "seam_bins": {"Full": [-4, 5], "Length": [5, 7], "Short": [7, 10], "Bouncer": [10, 16]},
        "spin_bins": {"OP": [-2, 2.8], "Full": [2.8, 4.4], "Good": [4.4, 6.2], "Short": [6.2, 15.0]}
    },
    "Men's ODI": {
        "ball_type": "White",
        "seam_bins": {"Full Toss": [-4, 0.5], "Yorker": [0.5, 2.5], "The Slot": [2.5, 5.8], "Length": [5.8, 8], "Short": [8, 10], "Bouncer": [10, 16]},
        "spin_bins": {"OP": [-2, 2.8], "Full": [2.8, 4.4], "Good": [4.4, 6.2], "Short": [6.2, 15.0]}
    },
    "Women's ODI": {
        "ball_type": "White",
        "seam_bins": {"Full Toss": [-5, 0.9], "Yorker": [0.9, 2.8], "The Slot": [2.8, 5.5], "Length": [5.5, 8], "Short": [8, 16]},
        "spin_bins": {"OP": [-2, 2.5], "Full": [2.5, 4], "Good": [4, 5.7], "Short": [5.7, 12.0]}
    }
}
