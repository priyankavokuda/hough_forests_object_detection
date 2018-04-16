#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 17:21:12 2017

@author: priyanka
"""
import param

class Rectangle():
    def __init__(self,**kwargs):
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.conf = 0

class Hypothesis():
    
    def __init__(self):
        
        self.conf = 0
        self.param = param.Parameter()
        self.bb = []
        self.truePos = []
        self.falsePos = []
        self.falseNeg = []
        for i in range(self.param.max_detections):
            self.bb.append(Rectangle())


#h = Hypothesis()
#h.readMedianHW()