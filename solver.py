import time
import numpy as np
import cv2 as cv
from mss import MSS
# import mss.tools
from pathlib import Path
from collections import namedtuple

#numbers and symbols folder paths
nums_folder = Path("numbers")
syms_folder = Path("symbols")

#test image
sample_problem = Path("sampleproblems/11x64.png")

#use namedtuple objects to later pass into nms to settle overlaps
instance = namedtuple('Instance', ["numsym", "score", "x1", "y1", "x2", "y2"])

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

    #grayscale and then binary (b/w)
    ss_array_gray = cv.cvtColor(ss_array, cv.COLOR_BGR2GRAY)
    
    return ss_array_gray

def iou(instances_list, iou_output_instances = []): #Intersection over Union (kinda)
    if len(instances_list) <= 1:
        iou_output_instances.append(instances_list[0])
        return iou_output_instances
    best = max(instances_list, key=lambda i: i.score) #find best instance
    best_center = (best.x1 + best.x2)/2
    other_instances = []
    for inst in instances_list:
        if not(best_center <= inst.x2 and best_center >= inst.x1):
            other_instances.append(inst) #add instances that dont have the best's center within their bounding box
    iou_output_instances.append(best) #add best to master list
    if len(other_instances) != 0:
        iou(other_instances, iou_output_instances) #if there are extras, recurision to iron out duplicates
    else:
        return iou_output_instances

def nms(template_h, template_w, label, array, thresh):
    #use noise max suppression (NMS) = remove the double counting of #s causing semantic error
    locs = np.where(array >= thresh)
    instances = [] #make list of instances
    for x1, y1 in zip(*locs[::-1]):
        score = array[y1, x1]
        inst = instance(label, score, x1, y1, x2 = x1 + template_w, y2 = y1 + template_h)
        instances.append(inst)
    post_iou_instances_list = iou(instances)
    
    return post_iou_instances_list #list of objects that have been through iou, so position is final

def graytempmatch(grayarray): #template matching to find numbers and symbols in equation
    #numbers:
    for img in nums_folder.glob("*.png"):
        template = cv.imread(img, cv.IMREAD_GRAYSCALE)
        numtemp_h, numtemp_w = template.shape[:2]
        current_num = img.name[3]
        num_res = cv.matchTemplate(grayarray, template, cv.TM_CCOEFF_NORMED)
        post_iou_num_instances = nms(numtemp_h, numtemp_w, current_num, num_res, 0.96)

    #symbols:
    name_to_sym = {"symplus.png" : "+", "symsub.png" : "-", "symmult.png" : "*", "symdiv.png" : "/"}
    
    for img in syms_folder.glob("*.png"):
        template = cv.imread(img, cv.IMREAD_GRAYSCALE)
        symtemp_h, symtemp_w = template.shape[:2]
        current_sym = name_to_sym[img.name]
        sym_res = cv.matchTemplate(grayarray, template, cv.TM_CCOEFF_NORMED)
        post_iou_sym_instances = nms(symtemp_h, symtemp_w, current_sym, sym_res, 0.96)

    return post_iou_num_instances, post_iou_sym_instances

def puttingitalltogether(num_instances, sym_instances):
    expression_dict = {}
    
    for inst in num_instances, sym_instances:
        expression_dict[inst.x1] = inst.numsym

    #sort into full expression and turn into string
    sorted_expression = dict(sorted(expression_dict.items()))
    expression_str = "".join(sorted_expression.values())
    return expression_str

#main loop
problem = pracprob(sample_problem)
number_instances, symbol_instances = graytempmatch(problem)
fullexpression = puttingitalltogether(number_instances, symbol_instances)
answer = eval(fullexpression) #compute answer
print(answer)