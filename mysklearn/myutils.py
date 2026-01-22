# import numpy as np

def my_discretizer(y):
    '''
        Purpose: maps numeric predictions to strings: "high" or "low".
    '''
    if y >= 100:
        return "high"
    else:
        return "low"
    

def mpg_discretizer(y):
    '''
        Purpose: maps continuous mpg predictions to a DOE category 1-10.
    '''
    # assign categorical categories (1-10) based on its continuous mpg value. 
    if y <=13:
        return 1
    elif 13 <= y <= 14:
        return 2
    elif 14 < y <= 16:
        return 3
    elif 16 < y <= 19:
        return 4
    elif 19 < y <= 23:
        return 5
    elif 23 < y <= 26:
        return 6
    elif 26 < y <= 30:
        return 7
    elif 30 < y <= 36:
        return 8
    elif 36 < y <= 44:
        return 9
    else:
        return 10