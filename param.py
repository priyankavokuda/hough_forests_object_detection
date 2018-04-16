#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 18:20:12 2017

@author: priyanka
"""

from configparser import ConfigParser

class Parameter(): 
    
    def __init__(self):
        self.config = ConfigParser()
        self.config.read_file(open('configuration.txt'))
        
        self.abs_path = self.config.get('path', 'abs_path')
        
        self.image_path = self.config.get('path', 'images_path')
        self.image_path = self.abs_path + self.config.get('path', 'images_path')
        
        self.train_annotation_path = self.config.get('path', 'train_annotation_file')
        self.train_annotation_path = self.abs_path + self.train_annotation_path
        
        self.test_annotation_path = self.config.get('path', 'test_annotation_file')
        self.test_annotation_path = self.abs_path + self.test_annotation_path
        
        self.forest_obj_file = self.config.get('path','forest_obj_file')

        self.patch_width = int(self.config.get('patch', 'patch_width'))
        self.patch_height = int(self.config.get('patch', 'patch_height'))
        self.num_patches = int(self.config.get('patch', 'num_patches'))
        self.ch = int(self.config.get('patch', 'ch'))
        
        self.num_files_train = int(self.config.get('tree','num_files_train'))
        self.num_trees = int(self.config.get('tree', 'num_trees'))
        self.max_depth = int(self.config.get('tree','max_depth'))
        self.min_sample_leaf = int(self.config.get('tree','min_sample_leaf'))
        self.num_threshold = int(self.config.get('tree','num_threshold'))
        self.min_pfg_vote = int(self.config.get('tree','min_pfg_vote'))
        self.max_votes_leaf = int(self.config.get('tree','max_votes_leaf'))
        
        self.inf_margin = float(self.config.get('training','inf_margin'))
        self.gain_margin = float(self.config.get('training','gain_margin'))
        self.num_tests = int(self.config.get('training','num_tests'))
        self.epsilon  = float(self.config.get('training','epsilon'))
        self.gt_bbs_filename = self.config.get('training','gt_bbs_filename')
        
        self.max_detections  = int(self.config.get('testing','max_detections'))
        self.test_param_file_name = self.config.get('testing','test_param_file_name')
        self.clear_area_h = int(self.config.get('testing','clear_area_h'))
        self.clear_area_w = int(self.config.get('testing','clear_area_w'))
#        print (self.max_detections,self.test_param_file_name)

#
#
#p=Parameter()
#p.initParameters()