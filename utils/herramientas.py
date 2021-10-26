from os import scandir
import csv
import pandas as pd
import numpy as np
import colorsys
import random
import matplotlib.pyplot as plt

"""
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color    
    Label(  'traffic light'  [(250,170, 30) )],
    Label(  'traffic sign'   [(220,220,  0) )],
    Label(  'person'         [(220, 20, 60) )],
    Label(  'rider'          [(255,  0,  0) )],
    Label(  'car'            [(  0,  0,142) )],
    Label(  'truck'          [(  0,  0, 70) )],
    Label(  'bus'            [(  0, 60,100) )],
    Label(  'train'          [(  0, 80,100) )],
    Label(  'motorcycle'     [(  0,  0,230) )],
    Label(  'bicycle'        [(119, 11, 32) )],
]
"""
def colores():
    colors=[(250,170, 30)
            ,(220,220,  0) 
            ,(220, 20, 60) 
            ,(255,  0,  0) 
            ,(  0,  0,142) 
            ,(  0,  0, 70) 
            ,(  0, 60,100) 
            ,(  0, 80,100) 
            ,(  0,  0,230) 
            ,(119, 11, 32)]
    return colors

det_class=['traffic sign','bicycle','person','car','traffic light','truck','rider','motorcycle','bus']