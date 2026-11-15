import time
import numpy as np
import cv2 as cv
from mss import MSS
# import mss.tools
from pathlib import Path

#numbers and symbols folder paths
nums_folder = Path("numbers")
syms_folder = Path("symbols")

def pracprob(img_path):
    pp_array_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    return pp_array_gray

def screenie(top = 363, left = 567, width = 150, height = 70):
    #grab the screenshot
    with MSS() as sct:
        bounding_box = {'top': top, 'left': left, 'width': width, 'height': height}
        screenshot = sct.grab(bounding_box)

    #convert ss to numpy array
    ss_array = np.array(screenshot)
    # print(ss_array)

    #grayscale and then binary (b/w)
    ss_array_gray = cv.cvtColor(ss_array, cv.COLOR_BGR2GRAY)
    return ss_array_gray
    # thresh1, ss_array_binary = cv.threshold(ss_array_gray, 127, 255, cv.THRESH_BINARY) #use grayscale instead for now

def graytempmatch(grayarray):
    #template matching to find numbers and symbols
    #numbers:
    num_locs = {} #dictionary that has number keys (str) and location values (list of tuples)
    for img in nums_folder.iterdir():
        template = cv.imread(img, cv.IMREAD_GRAYSCALE)
        current_num = img.name[3]
        num_res = cv.matchTemplate(grayarray, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(num_res >= threshold)
        num_locs[current_num] = [pt for pt in zip(*loc[::-1])]

    #symbols:
    sym_certainty = {} #dictionary that has symbol keys (str) and max values (floats)
    sym_best_locations = {} #dictionary of symbol keys (str) and locations (tuples) of best certainty value
    for img in syms_folder.iterdir():
        template = cv.imread(img, cv.IMREAD_GRAYSCALE)
        if img.name == "symplus.png": current_sym = "+"
        elif img.name == "symsub.png": current_sym = "-"
        elif img.name == "symmult.png": current_sym = "*"
        elif img.name == "symdiv.png": current_sym = "/"
        sym_res = cv.matchTemplate(grayarray, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(sym_res)
        sym_certainty[current_sym] = max_val
        sym_best_locations[current_sym] = max_loc

    return num_locs, sym_certainty, sym_best_locations

def puttingitalltogether(number_locations, symbol_certainties, symbol_best_locations):
    #initalize full expression dictionary with keys x positions and values number/symbol
    expression_dict = {}

    #best guess at which symbol it is
    best_symbol = max(symbol_certainties, key=symbol_certainties.get)
    #add position and symbol to full expression dict
    best_symbol_pos = int(symbol_best_locations[best_symbol][0])
    expression_dict[best_symbol_pos] = best_symbol

    #find locations of number instances and add to full expression dictionary
    for num in number_locations:
        for instance in number_locations[num]:
            inst_pos = int(instance[0]) # x position of current number instance
            expression_dict[inst_pos] = num

    #sort into full expression and turn into string
    sorted_expression = dict(sorted(expression_dict.items()))
    expression_str = "".join(sorted_expression.values())
    return expression_str

#test image
sample_problem = Path("sampleproblems/11x64.png")

#main loop
arrayfortempmatch = pracprob(sample_problem)
num_locs, sym_certainty, sym_best_locations = graytempmatch(arrayfortempmatch)
fullexpression = puttingitalltogether(num_locs, sym_certainty, sym_best_locations)
answer = eval(fullexpression) #compute answer