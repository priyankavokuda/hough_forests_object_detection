#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:52:12 2017

@author: priyanka

"""
import random
import param

class BinaryTest():
    def __init__(self):
        self.param = param.Parameter()
        self.x1=0
        self.x2=0
        self.y1=0
        self.y2=0
        self.ch=0
        self.tao=0
    
    def generate(self):
        self.x1 = random.randint(0,self.param.patch_width-1)
        self.y1 = random.randint(0,self.param.patch_height-1) 
        self.x2 = random.randint(0,self.param.patch_width-1)
        self.y2 = random.randint(0,self.param.patch_height-1) 
        self.ch = random.randint(0,self.param.ch-1)
#        return x1,y1,x2,y2,ch
    
    def evaluateValue(self, patch):
        return float(patch[self.x1,self.y1,self.ch] - float(patch[self.x2,self.y2,self.ch]))
    
    def change_tao(self,t):
        self.tao=t
    
    def detectEvaluate(self,patch):
        diff = float(patch[self.x1,self.y1,self.ch]) - float(patch[self.x2,self.y2,self.ch])
        return diff > self.tao