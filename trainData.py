#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 21:52:39 2017

@author: priyanka
"""


from PIL import Image
#from matplotlib import pyplot as plt
from sklearn.feature_extraction import image
import numpy as np
import random
import statistics
import sys
import time
import os

import annot
import param

sys.path.insert(0, "C:/master_thesis/code/feat/")
import featextract



def getIndicesAndOffset(img_patch,patch_x,patch_y,bb_center):
    """ get top left indices of patches and offset from bounding box """
    dist = [] 
    xx, yy = np.meshgrid(range(img_patch.shape[1]-(patch_y-1)), range(img_patch.shape[0]-(patch_x-1)))
    indices= np.hstack( [yy.reshape(-1,1), xx.reshape(-1,1)] )
#    print("bb_center",bb_center)
    if bb_center!=None:
        dist =  (indices + np.array([patch_y/2, patch_x/2])) - np.array(bb_center)   
    else:
        dist.append([None,None])
    return indices.tolist(),dist

def getMedianHW(heights,widths):
    p = param.Parameter()
    m_height = statistics.median(heights)
    m_width = statistics.median(widths)
    file_obj = open(p.test_param_file_name,'w')
    file_obj.write(str(m_width))
    file_obj.write('\n')
    file_obj.write(str(m_height))
    file_obj.close()
    
def isInside( x,  y,  bb_x, bb_y, bb_w, bb_h):
    """ check if point (x,y) is inside rerctangle bb """
    return (x >= bb_x and y >= bb_y and x < bb_x + bb_w and y < bb_y + bb_h)

def intersects(gt_box, d_box):
#    // If one rectangle is on left side of other
    if(d_box[1] > gt_box[0]+gt_box[2] or gt_box[0] > d_box[1]+d_box[2]):
        return False
#    // If one rectangle is above other
    if(d_box[0] > gt_box[1]+gt_box[3] or gt_box[1] > d_box[0]+d_box[3]):
        return False
    return True

#def plot(patches):
#    """ plot patches """
#    fig, ax = plt.subplots(nrows=10, ncols=10)
##    for i in range(1,len(patches)):
#    for i in range(1,100):
#        ax=plt.subplot(10, 10, i)
#        img=np.asarray(patches[i])
#        ax.imshow(img)
#    plt.show()
    
class PatchFeature():
    """ patch feature class """
    def __init__(self):
        self.patches=[]
        self.offset=[]

class TrainingData():
    
    def __init__(self):
        self.para = param.Parameter()
        train_fileName = self.para.train_annotation_path
        self.anno = annot.Annotation(train_fileName)
        
        self.p_width=self.para.patch_width
        self.p_height=self.para.patch_height
#        self.num_files_train = self.para.num_files_train
        self.pos_patch_feature = PatchFeature()
        self.neg_patch_feature = PatchFeature()
        self.feature = featextract.FeatureExtractorRaw()
        self.ch = self.para.ch
        self.bb_dict = {}#num of files to train is arg
        self.num_pos_patches_per_bb = 0
        self.num_neg_patches_per_img = 0
        

    def sampleInside(self):
        """ get pos patches """
#        start = time.clock()
        width_list = []
        height_list = []
        self.bb_dict = self.anno.getBoundingBox()
        self.num_pos_patches_per_bb,self.num_neg_patches_per_img=self.anno.numTotalPatches()
        for filename,bbs in self.bb_dict.items():
            path = os.path.join(self.para.image_path,str(filename))
#            img = Image.open(path+'.jpg')
#            print(path)
            img = Image.open(path)
##           get features 
            img_feature = self.feature.get_features(np.asarray(img)[...,:self.ch ])
            for bb in bbs:#for all bbs in image at 'path'
                x = bb[0]
                y = bb[1]
                w = bb[2]
                h = bb[3]
                width_list.append(w)
                height_list.append(h)
                crp_img = img_feature[y: y+h, x: x+w]
                crp_img = np.asarray(crp_img)
                patches = image.extract_patches_2d(crp_img, (self.p_width, self.p_height))
                x_center = (h/2)
                y_center = (w/2)
                bb_center=[x_center,y_center]
                indices_list,offset=getIndicesAndOffset(crp_img,self.p_width,self.p_height,bb_center)

                rand_x=np.random.randint(0,h-self.p_width,size=self.num_pos_patches_per_bb)
                rand_y=np.random.randint(0,w-self.p_height,size=self.num_pos_patches_per_bb)

                self.pos_patch_feature.patches.extend( [patches[i] for i in [indices_list.index([x,y]) for x,y in zip(rand_x,rand_y)]])
                self.pos_patch_feature.offset.extend( [offset[i] for i in [indices_list.index([x,y]) for x,y in zip(rand_x,rand_y)]])

        getMedianHW(height_list,width_list)
#        print ("pos patches",len(self.pos_patch_feature.patches))
#        print ("pos offset",len(self.pos_patch_feature.offset))
#        print("time taken for pos patch extrct",time.clock() - start,"secs")
        return self.pos_patch_feature
 
    def sampleOutside(self):
        """ get neg patches """
#        start = time.clock()
        cnt = 0
        for filename,bbs in self.bb_dict.items():
            path = self.para.image_path+'/'+str(filename)
#            img = Image.open(path+'.jpg')
            img = Image.open(path)
            img_feature = self.feature.get_features(np.asarray(img)[...,:self.ch ])
            patches = image.extract_patches_2d(img_feature, (self.p_width, self.p_height))
            all_ind_list,offset = getIndicesAndOffset(img_feature,self.p_width, self.p_height,None)
            cnt=self.num_neg_patches_per_img
#            start1 = time.clock()
            while(cnt>=0):
                gt_bb =[]
                x=random.randint(0,img_feature.shape[0]-self.p_height)
                y=random.randint(0,img_feature.shape[1]-self.p_width)
                gt_bb.append(x)
                gt_bb.append(y)
                gt_bb.append(x+self.p_width)
                gt_bb.append(x+self.p_height)
                rnd_pos_list=[]
                for bb in bbs:
#                    if isInside(x,y,bb[0],bb[1], bb[2], bb[3]):
                    if intersects(gt_bb,bb):   
#                        print("inside")
                        continue
                    else:
                        rnd_pos_list.append([x,y])
                        cnt=cnt-1
                        self.neg_patch_feature.patches.extend( [patches[i] for i in [all_ind_list.index([x,y])]])
#            print("time taken for while loop",time.clock() - start1,"secs")
        self.neg_patch_feature.offset = [None]*len(self.neg_patch_feature.patches)
#        print("time taken for neg patch extrct",time.clock() - start,"secs")
        return self.neg_patch_feature
                   
#t = TrainingData()
#pos_patches=t.sampleInside()
#neg_patches=t.sampleOutside()
#plot(pos_patches.patches)
#plot(neg_patches.patches)

#for i in range(len(pos_patches)):
#    img=np.asarray(pos_patches[i])
#    print(img.shape)
#    plt.imshow(img)
#
#    plt.show()
