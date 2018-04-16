#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:19:19 2017

@author: priyanka
"""

import param
import hfForest
import hypoth

import pickle
import time




class TrainDetect():
    def __init__(self):
        self.nTrees = param.Parameter().num_trees
        self.obj_fileName = param.Parameter().forest_obj_file

def train():
    forest = hfForest.HFForest()
    td = TrainDetect()
    for i in range(int(td.nTrees)):
        print()
        print("**** Training",i+1," tree ****")
        print()
        forest.train(i)

    forest_obj_file = open(td.obj_fileName, 'wb') 
    pickle.dump(forest, forest_obj_file,pickle.HIGHEST_PROTOCOL)
    print()
    print("**** TRAINING OVER ****")
    print()

def detect(tFile):          
    td = TrainDetect()
    obj_fileName = open(td.obj_fileName,'rb')
    forest = pickle.load(obj_fileName) 
    hyp = hypoth.Hypothesis()
    bbs,im_score = forest.detect(tFile,hyp)
    return bbs,tFile,im_score


if __name__ == '__main__':
     test_files = ['9.png','10.png']
     precision_list = []
     recall_list = []
     
     start = time.clock()
     train()
     print("Time take for training",time.clock() - start,"secs")
#     
     for tfile in test_files:
         start = time.clock()
         bbs,img_fileName,im_score = detect(tfile) #uncomment to detect
         print("Time take for testing",time.clock() - start,"secs")


    
        