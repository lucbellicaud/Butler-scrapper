from dataclasses import dataclass
from typing import Set
import numpy as np
import pandas as pd
import math


def calculate_difference(score1, score2):
    difference = score1 - score2
    
    if difference > 0:
        return map_difference_to_IMP(difference)
    else:
        return -map_difference_to_IMP(-difference)

def map_difference_to_IMP(difference):
    if difference <= 10:
        return 0
    elif difference >= 20 and difference <= 40:
        return 1
    elif difference >= 50 and difference <= 80:
        return 2
    elif difference >= 90 and difference <= 120:
        return 3
    elif difference >= 130 and difference <= 160:
        return 4
    elif difference >= 170 and difference <= 210:
        return 5
    elif difference >= 220 and difference <= 260:
        return 6
    elif difference >= 270 and difference <= 310:
        return 7
    elif difference >= 320 and difference <= 360:
        return 8
    elif difference >= 370 and difference <= 420:
        return 9
    elif difference >= 430 and difference <= 490:
        return 10
    elif difference >= 500 and difference <= 590:
        return 11
    elif difference >= 600 and difference <= 740:
        return 12
    elif difference >= 750 and difference <= 890:
        return 13
    elif difference >= 900 and difference <= 1090:
        return 14
    elif difference >= 1100 and difference <= 1290:
        return 15
    elif difference >= 1300 and difference <= 1490:
        return 16
    elif difference >= 1500 and difference <= 1740:
        return 17
    elif difference >= 1750 and difference <= 1990:
        return 18
    elif difference >= 2000 and difference <= 2240:
        return 19
    elif difference >= 2250 and difference <= 2490:
        return 20
    elif difference >= 2500 and difference <= 2990:
        return 21
    elif difference >= 3000 and difference <= 3490:
        return 22
    elif difference >= 3500 and difference <= 3990:
        return 23
    elif difference >= 4000:
        return 24
    else:
        return "error"
    
def IMP(score1, score2):
    if score1 == np.nan or score2 == np.nan:
        return np.nan
    else:
        difference = score1 - score2
    
        if difference > 0:
            return map_difference_to_IMP(difference)
        else:
            return -map_difference_to_IMP(-difference)

def round_to_nearest_10(number):
    if number >= 0:
        return (number // 10) * 10
    else:
        return ((number // 10) + 1) * 10

@dataclass
class MatchInfo :
    table_number : int
    home_team : str
    away_team : str
    pair_ns_open : Set[str]
    pair_ns_close : Set[str]
    pair_ew_open : Set[str]
    pair_ew_close : Set[str]