#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:46:42 2017

@author: priyanka
"""
import binaryTest
import param
import trainData
#from matplotlib import pyplot as plt
#import numpy as np
#from PIL import Image


class Node():
    def __init__(self):
        self.parent = None
        self.left_child = None
        self.right_child = None
        self.depth = 0
        self.branch = 0 #left or right child

class LeafNode(Node):
    def __init__(self):
        Node.__init__(self)

class TestNode(Node):
    def __init__(self):
        Node.__init__(self)
        self.binary_test = binaryTest.BinaryTest()

class InterNode(LeafNode):
    def __init__(self):
        LeafNode.__init__(self)
        self.param = param.Parameter()
        self.train_neg=trainData.PatchFeature()
        self.train_pos=trainData.PatchFeature()
        self.tests = []
        self.test_val_pos = []
        self.test_val_neg = []
        for i in range(self.param.num_tests):
            self.tests.append(binaryTest.BinaryTest())
        self.measureMode = 0
        self.pfg = 0
        self.list_centers = []
        self.current_value = 0
     
#    def plot(self,patches):
#        """ plot patches """
#        fig=plt.figure(figsize=(8, 8))
#        
#        columns = 10
#        rows = 10 
#        for i in range(0,min(len(patches),100)):
#            img=np.asarray(patches[i])
#            fig.add_subplot(rows, columns, i+1)
#            plt.imshow(img)
#        plt.show()
                
        
    def addTrainingdata(self,pos_patches,neg_patches):
        self.train_pos = pos_patches
        self.train_neg = neg_patches
        self.depth =0
        return self.train_pos,self.train_neg
    
    def makeLeaf(self,node,pov_ratio):
        points = []
        for pt in self.train_pos.offset:
            points.append(list(pt))
        node.pfg=len(self.train_pos.patches)/((pov_ratio*len(self.train_neg.patches))+len(self.train_pos.patches))#probability foreground
        node.num_neg = len(self.train_neg.patches)
        node.list_centers = self.train_pos.offset
        node.depth = self.depth
        node.branch = self.branch
        return node

        
#    def plotOffsets(self,points):
##    points = 
#        bbCenter=[77.0, 121.0]
#        plt.plot(bbCenter[0],bbCenter[1],'ro')
#        for pt in points:
#            plt.plot(bbCenter[0]+pt[0],bbCenter[1]+pt[1],'g*')
        
    

#points = [[ -2.,  23.], [ 27., -66.], [ 28.,  52.], [-37., -32.], [ 45.,  53.]]
#plotOffsets(points,'b')
#points = [[-42.,  41.], [-40.,  25.], [-44., -63.], [-39., -41.]]
#plotOffsets(points,'g')
#plt.show()