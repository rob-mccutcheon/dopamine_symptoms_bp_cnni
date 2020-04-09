#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 13:15:09 2017

@author: rob mccutcheon
"""
from configparser import ConfigParser
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from json import loads
import sys
import ast
import os

# N.B. node_ids are indexed from 0, but network ids start from 1
# Load directory and file ids
parser = ConfigParser()
#  Amend parser.read statement and amend project_details.txt file depending on where you are running
parser.read('project_details.txt')
node_selection=sys.argv[2]
main_directory = parser.get('project_details', 'main_directory')
beta_directory = parser.get('project_details', 'beta_directory')
roi_directory = parser.get('project_details', 'roi_directory')
voxel_maps_dir = ('%sresults/voxel_assigned_maps' % main_directory)
voxel_probability_maps_dir = (f'{main_directory}results/voxel_probability_maps/{node_selection}')
node_id_file = parser.get('project_details', 'node_id_file')
striatum_file = parser.get('project_details', 'striatum_file')
num_networks = int(parser.get('project_details', 'num_networks'))
num_top_seeds = int(parser.get('project_details', 'num_top_seeds'))
img_dim = (loads(parser.get("project_details", "image_dimension")))
img_dim = [int(i)for i in img_dim]

#
if not os.path.exists(voxel_probability_maps_dir):
    os.mkdir(voxel_probability_maps_dir)


# Takes the subject id from the bash command line queue script
subject_id = sys.argv[1]

xvoxels = range(0, img_dim[0])
yvoxels = range(0, img_dim[1])
zvoxels = range(0, img_dim[2])

# Node id dictionary
apriori_communities = parser._sections[f'{node_selection}']
dict1 = {}
for k, v in apriori_communities.items():
    val_ints = ast.literal_eval(v)  # convert string to list
    dict1[k] = [x-1 for x in val_ints]  # subtract 1 frome each node_id

switched_dict = {str(val): key for (key, val) in dict1.items()}  # switch keys and value (convert back to string)

replacement_values = {'default': 1, 'smhand': 2, 'smmouth': 3, 'visual': 4, 'frontoparietal': 5, 'cinguloparietal': 6,
                      'retrosplenialtemporal': 7, 'cinguloperc': 8, 'dorsalattn': 9, 'auditory': 10, 'ventralattn': 11,
                      'salience': 12, 'none': 13}
replaced_dict = {k: replacement_values.get(v, v) for k, v in switched_dict.items()}

node_details = {}
for k, v in replaced_dict.items():
    key_ints = ast.literal_eval(k)
    for i in key_ints:
        node_details[i] = v


# all beta files for subject. File list runs through subject ids submitted by queue script
#filelist_init = sorted(glob.glob('%(1)s/BETA_Subject%(2)s*.nii' % {"1": beta_directory, "2": subject_id})) #original
filelist_init = sorted(glob.glob(f'{beta_directory}BETA_Subject{subject_id}*.nii')) #hcp
# the initial betas  those of eg movement regressors, we want the last ones
#filelist = filelist_init[-333:]#orig
filelist = filelist_init[0:333:]
betas = np.array([np.array((nib.load(fname, mmap=False)).get_data()) for fname in filelist])
a=[1,2,3,4,5]
a
a[0:2:]
# Network Probabilty Maps
mean_network_connectivity = np.zeros(img_dim+[num_networks])
for network in range(1, num_networks+1):
    nodes_in_network = [k for k, v in node_details.items() if float(v) == network]
    mean_network_connectivity[:, :, :, network-1] = np.mean(betas[nodes_in_network, :, :, :], axis=0)

# NB all negative connectivity weights set to zero
pos = np.where(mean_network_connectivity > 0, mean_network_connectivity, 0)

# make each voxel add up to 1 when all networks combined
network_probability_map = np.zeros(img_dim+[num_networks])
for network in range(num_networks):
    network_probability_map[:, :, :, network] = pos[:, :, :, network]/np.sum(pos, axis=3)
np.save('%(1)s/voxel_probabilities_%(2)s.npy' % {"1": voxel_probability_maps_dir, "2": subject_id},
        network_probability_map)
