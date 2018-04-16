#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 10:32:01 2017

@author: priyanka
"""


import pandas as pd
import param
import ast
from collections import OrderedDict

import cgitb
cgitb.enable()


class Annotation():
    def __init__(self,annotation_path):
        self.para = param.Parameter()
        reader = pd.read_csv(annotation_path)
        self.files = reader['#filename']
        self.bbs = reader['region_shape_attributes']
        self.bb_dict=OrderedDict()
     
    def getBoundingBox(self,*args):
        """ get bounding boxes from images """
        try: 
            self.num_images = args[0]
        except:
            self.num_images = len(self.files)
        for i in range(self.num_images):
            x = (ast.literal_eval(self.bbs[i]).get('x'))# convert to dict
            y = (ast.literal_eval(self.bbs[i]).get('y'))
            width = (ast.literal_eval(self.bbs[i]).get('width'))
            height = (ast.literal_eval(self.bbs[i]).get('height'))
            bb = [x,y,width,height]
            if self.files[i] in self.bb_dict.keys():
                self.bb_dict[self.files[i]].append(bb)
            else:
                self.bb_dict[self.files[i]] = [bb]
        return self.bb_dict
    
    
    def numTotalPatches(self):
        """ calculate total num of patches per bb for pos features and per image for neg features """
        num_patches=int(self.para.num_patches)
        
        total_imgs=0
        total_bbs=0
        
        for keys,values in self.bb_dict.items():
            total_imgs+=1
            total_bbs+=len(values)
        print("total images",total_imgs,"total bbs",total_bbs)
        if num_patches > total_imgs:
            self.num_neg_patches_per_img = round(num_patches/total_imgs)
        else:
            print("Number of patches ("+str(num_patches)+") should be greater than number of images ("+str(total_imgs)+") change in configuration.txt")
            return     
        if num_patches > total_bbs:
            self.num_pos_patches_per_bb = round(num_patches/total_imgs)
        else:
            print("Number of patches ("+str(num_patches)+") should be greater than number of bounding boxes ("+str(total_bbs)+") change in configuration.txt")
            return
#        print ("pos patches per bb",self.num_pos_patches_per_bb,"neg patch per image",self.num_neg_patches_per_img)
        return self.num_pos_patches_per_bb,self.num_neg_patches_per_img
    

