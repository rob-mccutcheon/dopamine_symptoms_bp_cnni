#!/usr/bin/env python36
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
import random
import os

# N.B. node_ids are indexed from 0, but network ids start from 1
# Load directory and file ids
parser = ConfigParser()
#  Amend parser.read statement and amend project_details.txt file depending on where you are running
parser.read('project_details.txt')
#parser.read('/Users/robmcc/mnt/droplet/home/k1201869/DOPA_symptoms/src/project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
beta_directory = parser.get('project_details', 'beta_directory')
roi_directory = parser.get('project_details', 'roi_directory')
voxel_maps_dir = ('%sresults/voxel_assigned_maps' % main_directory)
node_id_file = parser.get('project_details', 'node_id_file')
striatum_file = parser.get('project_details', 'striatum_file')
pet_maps_dir = ('%sdata/ki_maps' % main_directory)
num_networks = int(parser.get('project_details', 'num_networks'))
num_top_seeds = int(parser.get('project_details', 'num_top_seeds'))
img_dim = (loads(parser.get("project_details", "image_dimension")))
img_dim = [int(i)for i in img_dim]
network_key = parser._sections['network_key']

number_permutations = 10000
node_assignment = 'node_ids'
# Takes the subject id from the bash command line queue script
#  subject_id = '001'
subject_id = sys.argv[1]

# Node id dictionary
apriori_communities = parser._sections[node_assignment]
dict1 = {}
for k, v in apriori_communities.items():
    val_ints = ast.literal_eval(v)  # convert string to list
    dict1[k] = [x-1 for x in val_ints]  # subtract 1 from each node_id

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

# Load beta maps
# all beta files for subject. File list runs through subject ids submitted by queue script
filelist_init = sorted(glob.glob('%(1)s/BETA_Subject%(2)s*.nii' % {"1": beta_directory, "2": subject_id}))
# the initial betas maybe those of eg movement regressors, we want the last ones
filelist = filelist_init[-333:]
betas = np.array([np.array((nib.load(fname, mmap=False)).get_data()) for fname in filelist])

# Load striatal mask
striatal_mask_nii = (roi_directory+striatum_file)
striatal_mask_temp = nib.load(striatal_mask_nii, mmap=False).get_data()  # disables 'memmap' special arrays
striatal_mask = 1 <= striatal_mask_temp
striatal_mask_multidim = np.stack([striatal_mask]*num_networks, axis=-1)

# Load PET ki_map
pet_filename = os.listdir('%(1)s/%(2)s/' % {"1": pet_maps_dir, "2": subject_id})
pet_map_nii = nib.load('%(1)s/%(2)s/%(3)s' % {"1": pet_maps_dir, "2": subject_id, "3": pet_filename[0]})
pet_map = np.array(pet_map_nii.get_data())

# Array to store mean ki for each network
network_kis = np.zeros([number_permutations+1, num_networks])
network_kis[0, :] = range(1, num_networks+1)

perm_results_dir = f'{main_directory}/results/null_dist/cortical_perm/{node_assignment}/'
if not os.path.exists(perm_results_dir):
    os.mkdir(perm_results_dir)
i=0
for i in range(number_permutations):
    # shuffle node-network pairings
    values = list(node_details.values())
    random.shuffle(values)
    shuffled_node_details = dict(zip(node_details, values))

    # Network Probabilty Maps
    mean_network_connectivity = np.zeros(img_dim+[num_networks])
    for network in range(1, num_networks+1):
        nodes_in_network = [k for k, v in shuffled_node_details.items() if float(v) == network]
        mean_network_connectivity[:, :, :, network-1] = np.mean(betas[nodes_in_network, :, :, :], axis=0)

    pos = np.where(mean_network_connectivity > 0, mean_network_connectivity, 0)

    # make each voxel add up to 1 when all networks combined
    network_probability_map = np.zeros(img_dim+[num_networks])
    for network in range(num_networks):
        network_probability_map[:, :, :, network] = pos[:, :, :, network]/np.sum(pos, axis=3)

    # striatal masking
    striatum_only = np.zeros((img_dim+[num_networks]))
    striatum_only = striatal_mask_multidim*network_probability_map
    striatum_only = np.where(striatum_only == 0, np.nan, striatum_only)

    #remove 'none' and 'vis'
    preserved_networks = ['default', 'smhand', 'frontoparietal', 'auditory', 'dorsalattn', 'cinguloperc']
    preserved_indices = [network_key[k] for k in preserved_networks if k in network_key]
    preserved_indices = [int(i)-1 for i in preserved_indices]
    trimmed_map = striatum_only[:, :, :, preserved_indices]
    withoutvis = np.zeros(img_dim+[num_networks])
    withoutvis[:, :, :, preserved_indices] = trimmed_map
    new_denominator = np.stack([np.nansum(withoutvis, axis=3)]*num_networks, axis=-1)
    normalised_without_vis = withoutvis/new_denominator


    # Mean ki for each networks
    for network in range(1, num_networks+1):
        # Mask pet map for each network and weight by probabilities(nb - TRUE means the value is masked)
        probability_sum = np.nansum(normalised_without_vis[:, :, :, network-1])
        weighted_ki = np.multiply(normalised_without_vis[:, :, :, network-1], pet_map)
        network_kis[i+1, network-1] = np.nansum(weighted_ki)/probability_sum

np.save(f'{main_directory}results/null_dist/cortical_perm/{node_assignment}/network_kis_{subject_id}.npy', network_kis)
