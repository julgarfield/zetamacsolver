import time
import numpy as np
import cv2 as cv
from mss import MSS
# import mss.tools
from pathlib import Path

#numbers and symbols folder paths
nums_folder = Path("numbers")
syms_folder = Path("symbols")

# time.sleep(3.5)

#grab the screenshot
with MSS() as sct:
    bounding_box = {'top': 363, 'left': 567, 'width': 150, 'height': 70}
    screenshot = sct.grab(bounding_box)

#convert ss to numpy array
ss_array = np.array(screenshot)
# print(ss_array)

#grayscale and then binary (b/w)
ss_array_gray = cv.cvtColor(ss_array, cv.COLOR_BGR2GRAY)
thresh1, ss_array_binary = cv.threshold(ss_array, 127, 255, cv.THRESH_BINARY)

#template matching to find numbers and symbols
#numbers:
num_locs = {} #dictionary that has number keys (str) and location values (list of tuples)
for img in nums_folder:
    template = cv.imread(img, cv.IMREAD_GRAYSCALE)
    current_num = img.name[3]
    num_res = cv.matchTemplate(ss_array_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(num_res >= threshold)
    num_locs[current_num] = [pt for pt in zip(*loc[::-1])]

#symbols:
sym_certainty = {} #dictionary that has symbol keys (str) and max values (floats)
sym_best_locations = {} #dictionary of symbol keys (str) and locations (tuples) of best certainty value
for img in syms_folder:
    template = cv.imread(img, cv.IMREAD_GRAYSCALE)
    if img.name == "symplus.png": current_sym = "+"
    elif img.name == "symsub.png": current_sym = "-"
    elif img.name == "symmult.png": current_sym = "*"
    elif img.name == "symdiv.png": current_sym = "/"
    sym_res = cv.matchTemplate(ss_array_gray, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(sym_res)
    sym_certainty[current_sym] = max_val
    sym_best_locations[current_sym] = max_loc

#initalize full expression dictionary with keys x positions and values number/symbol
expression_dict = {}

#best guess at which symbol it is
best_symbol = max(sym_certainty, key=sym_certainty.get)
#add position and symbol to full expression dict
best_symbol_pos = int(sym_best_locations[best_symbol][0])
expression_dict[best_symbol_pos] = best_symbol

#find locations of number instances and add to full expression dictionary
for num in num_locs:
    for instance in num_locs[num]:
        inst_pos = int(num_locs[num][instance][0]) # x position of current number instance
        expression_dict[inst_pos] = num

#sort into full expression and turn into string
sorted_expression = dict(sorted(expression_dict.items()))
expression_str = "".join(sorted_expression.values())

#compute answer
answer = eval(expression_str)