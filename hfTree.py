#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:34:50 2017

@author: priyanka
"""
import param
import binaryTest
import node
import random
import numpy as np
import trainData
from matplotlib import pyplot as plt

random.seed(3)

class HFTree():
    def __init__(self):
        self.param = param.Parameter()
        self.leaves=[]
        self.nodes=[]
        
    def train(self):
        td = trainData.TrainingData()
        self.pos_patches = td.sampleInside()
        self.neg_patches = td.sampleOutside()
        self.root = node.InterNode()
        self.pov_ratio = (len(self.pos_patches.patches))/(len(self.neg_patches.patches))
        self.root.addTrainingdata(self.pos_patches,self.neg_patches)  
        print ("Pos",len(self.root.train_pos.patches)," and Neg patches",len(self.root.train_neg.patches)," added to root")
        self.grow(self.root,None)

                
    def grow(self,inode,parent):
        ''' add nodes to root '''
        if (inode.depth == self.param.max_depth-1):
            self.makeLeaf(inode,parent)
        else:
           if (len(inode.train_pos.patches) > 0 and (len(inode.train_pos.patches)+len(inode.train_neg.patches) > self.param.min_sample_leaf)):
               b_test = binaryTest.BinaryTest()
               left = node.InterNode()
               right = node.InterNode()
               b_test,right,left,valid = self.optimizeTest(inode,left,right,b_test)#measure distGain for regression test, InfGain for classification test
               if valid:
#                   print ("**valid splits")
                   self.makeTestNode(inode,parent,left,right,b_test)
               else:
#                   print ("**no valid splits")
                   self.makeLeaf(inode,parent) # Not valid splits
           else:
#                print ("**not enough samples")
                self.makeLeaf(inode,parent) # Not enough samples           
    
    def optimizeTest(self,current,left,right,b_test):
        ''' check for valid splits '''
        best_gain = 0
        found = False
        current.measure_mode = 0 # Regression
        if(len(current.train_pos.patches) < (0.95 *len(current.train_pos.patches)+len(current.train_neg.patches)) and (current.depth < self.param.max_depth-2)):# Last test nodes based on distGain
            current.measure_mode = random.randint(0, 2)
#        print("mode: ",current.measure_mode)
        current.current_value = self.measureNode(current,current.measure_mode)
        gain_margin = self.getGainMargin(current.measure_mode)
        
        for i in range(self.param.num_tests):
            test_threshold_list=[]
            current.tests[i].generate()#Generate two sets of random pixel values from patches 

            test_val_pos_list = [current.tests[i].evaluateValue(pp) for pp  in current.train_pos.patches]
            test_val_neg_list = [current.tests[i].evaluateValue(pn) for pn in current.train_neg.patches]

            current.test_val_pos.append(test_val_pos_list)
            current.test_val_neg.append(test_val_neg_list)
            
            max_test_val = max(test_val_pos_list)
            min_test_val = min(test_val_pos_list)
#            print("depth: ",current.depth)
            for j in range(self.param.num_threshold):
#                print (min_test_val, max_test_val)
                rand_threshold = random.randint(min_test_val,max_test_val)
#                print (randThreshold)
                test_threshold_list.append(rand_threshold)
                tmp_left=node.InterNode()
                tmp_right=node.InterNode()
                
                tmp_right,tmp_left = self.split(current, tmp_left, tmp_right, test_val_pos_list, test_val_neg_list, test_threshold_list[j])
                
                if((len(tmp_left.train_pos.patches)+len(tmp_left.train_neg.patches) > self.param.min_sample_leaf) and (len(tmp_right.train_pos.patches)+len(tmp_right.train_neg.patches) > self.param.min_sample_leaf)  ):
                   tmpGain = self.measureSplit(tmp_left, tmp_right, current.current_value, current.measure_mode)
                   if(tmpGain > gain_margin and tmpGain > best_gain): 
                    	    found = True
                    	    best_gain = tmpGain
                    	    b_test = current.tests[i]
                    	    b_test.change_tao(test_threshold_list[j])
                    	    left = tmp_left
                    	    left.depth = current.depth + 1
                    	    right = tmp_right
                    	    right.depth = current.depth + 1
        return b_test,right,left,found
    
    def measureNode(self,node,mode):
        if mode == 0:
            return self.entropy(node)
        else:
            return self.distMean(node)
        
    def entropy(self,node):
        p = len(node.train_pos.patches)/(self.pov_ratio * len(node.train_neg.patches)+len(node.train_pos.patches))
        entropy = 0
        if (p > 0 and p < 1):
            entropy = (- p*np.log(p) - (1-p)*np.log(1-p))
        return entropy
    
    def distMean(self,node):
        mean_x = 0.0
        mean_y = 0.0
        for i in range(len(node.train_pos.patches)):
            mean_x += node.train_pos.offset[i][0] #find mean
            mean_y += node.train_pos.offset[i][1]
        mean_x /= (len(node.train_pos.patches)+self.param.epsilon )
        mean_y /= (len(node.train_pos.patches)+self.param.epsilon )
    
        dist = 0
        for i in range(len(node.train_pos.patches)): #subtract mean
            dx = node.train_pos.offset[i][0] - mean_x
            dy = node.train_pos.offset[i][1] - mean_y  
            dist += ((dx*dx)+(dy*dy))   
        return np.sqrt(dist/(len(node.train_pos.patches)+self.param.epsilon ))
    
   
    def makeTestNode(self,inode,parent,left,right,test):
        tn = node.TestNode()
        tn.test =  test
        tn.depth = inode.depth
        tn.parent = parent
        self.nodes.append(tn)
        if(parent!=None): 
            if(inode.branch):
                parent.right_child = tn
            else:
                parent.left_child = tn
        left.branch = 0
#        print ("btest channel",test.ch)
        self.grow(left, tn)
        right.branch = 1
        self.grow(right, tn)
    
    
    def makeLeaf(self,inode,parent):
#        print ("creating leaf")
        leaf = node.LeafNode();
        leaf.parent = parent
        leaf.depth = parent.depth+1
        
        
        self.leaves.append(leaf)
        if(parent!=None): 
            if(inode.branch):
                parent.right_child = leaf
            else:
                parent.left_child = leaf
        leaf = inode.makeLeaf(leaf, self.pov_ratio)
    
    def getGainMargin(self,mode):
        if mode == 0:
            return self.param.inf_margin
        else:
            return self.param.gain_margin
    
    def plot(self,patches):
        """ plot patches """
        fig=plt.figure(figsize=(8, 8))
        
        columns = 10
        rows = 10 
        for i in range(0,min(len(patches),100)):
            img=np.asarray(patches[i])
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img)
        plt.show()
    
    def split(self,current, tmp_left, tmp_right, tests_val_pos, tests_val_neg, tao):
       ''' split patches according to threshold  ''' 
       neg_right_patches=[]
       neg_left_patches=[]
       pos_right_patches=[]
       pos_left_patches=[]
       for i in range(len(tests_val_pos)):
           if tests_val_pos[i] > tao:
               tmp_right.train_pos.patches.append(current.train_pos.patches[i])
               tmp_right.train_pos.offset.append(current.train_pos.offset[i])
               pos_right_patches.append(current.train_pos.patches[i])
#               self.plot(current.train_pos.patches[i])
           else:
               tmp_left.train_pos.patches.append(current.train_pos.patches[i])
               tmp_left.train_pos.offset.append(current.train_pos.offset[i])
               pos_left_patches.append(current.train_pos.patches[i])
#               self.plot(current.train_pos.patches[i])
               
       for i in range(len(tests_val_neg)):
           if tests_val_neg[i] > tao:
               tmp_right.train_neg.patches.append(current.train_neg.patches[i])
               tmp_right.train_neg.offset.append(current.train_neg.offset[i])
               neg_right_patches.append(current.train_neg.patches[i])
#               self.plot(current.train_neg.patches[i])
           else:
               tmp_left.train_neg.patches.append(current.train_neg.patches[i])
               tmp_left.train_neg.offset.append(current.train_neg.offset[i])
               neg_left_patches.append(current.train_neg.patches[i])
       return tmp_right,tmp_left
    
    def measureSplit(self,left,right,prev,mode):                 
        if (mode==0):
            return self.InfGain(left, right, prev) 
        else:
            return self.distGain(left,right,prev)
#
    def distGain(self,left,right,prev):
        ''' diff between entropy of parent and child nodes '''
        p_left = len(left.train_pos.patches)+(self.pov_ratio*len(left.train_neg.patches))
        p_right = len(right.train_pos.patches)+(self.pov_ratio*len(right.train_neg.patches))
        p_left = p_left/(p_left+p_right)
        p_right = 1 - p_left
        return prev - p_left*self.distMean(left) - p_right*self.distMean(right)
    
    def InfGain(self,left,right,prev):
        ''' diff between dist of parent and child nodes '''
        p_left = len(left.train_pos.patches) + (self.pov_ratio*len(left.train_neg.patches))
        p_right = len(right.train_pos.patches) + (self.pov_ratio*len(right.train_neg.patches))
        p_left = p_left/(p_right+p_left)
        p_right = 1 - p_left
        return prev - p_left*self.entropy(left) - p_right*self.entropy(right)
    
#    def recursive_inorder(self,root):
#        if not root:
#            return []
#        return self.recursive_inorder(root.leftChild) + [root.depth] + self.recursive_inorder(root.rightChild)

    def detection(self,patch):
        pnode = self.nodes[0]
        while pnode.left_child!=None: 
            if pnode.test.detectEvaluate(patch):
                pnode = pnode.right_child
            else:
                pnode = pnode.left_child
        return pnode

