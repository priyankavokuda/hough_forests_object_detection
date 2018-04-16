#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 23:33:50 2017

@author: priyanka
"""

import sys
import numpy as np
from sklearn.feature_extraction import image
from scipy.ndimage.filters import gaussian_filter
from matplotlib import pyplot as plt
from PIL import Image,ImageDraw

import hfTree
import trainData
import param
sys.path.insert(0, "C:/master_thesis/code/feat/")
import featextract


def getCentersOfPatches(img_patch,patch_x,patch_y):
    xx, yy = np.meshgrid(range(img_patch.shape[1]-(patch_y-1)), range(img_patch.shape[0]-(patch_x-1)))
    indices= np.hstack( [yy.reshape(-1,1), xx.reshape(-1,1)] )
    indices_list = indices+np.array([(patch_x)/2,(patch_y)/2])
    return indices_list

def plotDetections(hyp,src_img):
    c = 'blue'
    draw = ImageDraw.Draw(src_img)
    draw.line(((hyp.y,hyp.x),(hyp.y,hyp.x+hyp.h)),fill=c,width=10)
    draw.line(((hyp.y,hyp.x+hyp.h),(hyp.y+hyp.w,hyp.x+hyp.h)),fill=c,width=10)
    draw.line(((hyp.y+hyp.w,hyp.x+hyp.h),(hyp.y+hyp.w,hyp.x)),fill=c,width=10)
    draw.line(((hyp.y+hyp.w,hyp.x),(hyp.y,hyp.x)),fill=c,width=10)
    plt.imshow(src_img)
    
    
class HFForest():
    def __init__(self):
        self.param = param.Parameter()
        self.num_trees = self.param.num_trees
        self.image_path = self.param.image_path
        self.train_data = trainData.TrainingData()
        self.ch = self.param.ch
        self.list_trees = []
        for i in range(int(self.num_trees)):
            self.list_trees.append(hfTree.HFTree())
    
    def train(self,index):
        self.list_trees[index].train()
        print("Tree: num nodes",len(self.list_trees[index].nodes),"num leaves",len(self.list_trees[index].leaves))
        
    def detect(self,orig_img,hyp):
        orig_img = Image.open(self.image_path+'/'+orig_img)
        plt.imshow(orig_img)
        plt.title('Test image')
        plt.show()
        img = np.asarray(orig_img)[...,:self.ch ]
        vote_array = np.zeros((img.shape[0],img.shape[1]))
        img_feature = featextract.FeatureExtractorRaw().get_features(img) 
        vote_array = self.vote(vote_array,img_feature)
        bbs,im_score = self.detectFromVote(vote_array,orig_img,hyp)
        return bbs,im_score
    
    def vote(self,vote_array,img_feature):
        skip=8
        p_width =  self.param.patch_width
        p_height = self.param.patch_height
        patches = image.extract_patches_2d(img_feature, (p_width, p_height))
        centers = getCentersOfPatches(img_feature,p_width,p_height)
        patches = patches[0::skip]
        centers = centers[0::skip]
        for patch, center in zip(patches, centers):
            vote_array =  self.getVoteArray(vote_array,patch,center)  
        return vote_array

    
    def getVoteArray(self,vote_array,patch,center):
        result = self.detectLearn(patch)
        cx=center[0]
        cy=center[1]
        for inode in result:  
            if (inode.pfg > self.param.min_pfg_vote):
                weight = inode.pfg/(len(inode.list_centers)*len(result))
                wx = np.array([int(cx - x) for x in [cen[0] for cen in inode.list_centers]])
                wy = np.array([int(cy - y) for y in [cen[1] for cen in inode.list_centers]])
                for i in np.where((wx >= 0) & (wx < vote_array.shape[0]) & (wy >= 0) & (wy < vote_array.shape[1]))[0]:
                    vote_array[wx[i],wy[i]] += weight
        return vote_array
        
    
    def detectLearn(self, patch):
        result = []
        for i in range(len(self.list_trees)):      
            result.append(self.list_trees[i].detection(patch))
        return result
                    
    def detectFromVote(self,vote_array,img,hyp):
        blurred_vote_array = gaussian_filter(vote_array, sigma=7)
        plt.imshow(blurred_vote_array)
        plt.title('Debug image of vote array')
        np.save('votearray', blurred_vote_array)
        plt.show()
        im_score = -1
        file_obj = open(self.param.test_param_file_name, 'r') 
        lines = file_obj.readlines()
        for i in range(self.param.max_detections):
            max_conf = np.amax(vote_array)
            if max_conf > im_score:
                im_score = max_conf
            ind = np.where( vote_array == max_conf )
            x = ind[0][0]
            y = ind[1][0]
            hyp.bb[i].w = float(lines[0].strip('\n'))
            hyp.bb[i].h = float(lines[1].strip('\n'))
            hyp.bb[i].x = int(x - hyp.bb[i].h/2)
            hyp.bb[i].y =  int(y - hyp.bb[i].w/2)
            hyp.bb[i].conf = max_conf
            vote_array[x-int(self.param.clear_area_w/2):x+int(self.param.clear_area_w/2), y-int(self.param.clear_area_h/2):y+int(self.param.clear_area_h/2)] = 0
            plotDetections(hyp.bb[i],img)
        plt.title('Detections')
        plt.show()
        return hyp.bb,im_score
                        
            
        
        
            
            
    
        