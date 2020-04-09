from configparser import ConfigParser
import nibabel as nib
import numpy as np
import sys
from json import loads
from pathlib import Path
import os

# Load file details
parser = ConfigParser()
parser.read("/project_details.txt")
node_selection = sys.argv[2]
beta_directory = parser.get('project_details', 'beta_directory')
main_directory = parser.get('project_details', 'main_directory')
roi_directory = parser.get('project_details', 'roi_directory')
voxel_probability_maps_dir = f'{main_directory}results/voxel_probability_maps/{node_selection}' #orig
indiv_striatum_proability_maps_np_dir = f'{main_directory}results/indiv_striatum_probability_maps/{node_selection}/numpy' #orig
#indiv_striatum_proability_maps_np_dir = ('%sresults/hcp/indiv_striatum_probability_maps/data_driven_50nov1_2019/numpy' % main_directory) #hcp
indiv_striatum_proability_maps_nii_dir = f'{main_directory}/results/indiv_striatum_probability_maps/{node_selection}/nifti'  #orig
#indiv_striatum_proability_maps_nii_dir = ('%sresults/hcp/indiv_striatum_probability_maps/data_driven_50nov1_2019/nifti' % main_directory) #hcp
striatum_file = parser.get('project_details', 'striatum_file')
img_dim = (loads(parser.get("project_details", "image_dimension")))
num_networks = int(parser.get('project_details', 'num_networks'))

for path in [indiv_striatum_proability_maps_np_dir, indiv_striatum_proability_maps_nii_dir]:
    if not os.path.exists(path):
        Path(path).mkdir(parents=True)

# Load striatal mask
striatal_mask_nii = (roi_directory+striatum_file)
striatal_mask_temp = nib.load(striatal_mask_nii, mmap=False).get_data()  # disables 'memmap' special arrays
striatal_mask = 1 <= striatal_mask_temp

#get an affine
beta_sample = (nib.load('%sBETA_Subject002_Condition001_Source106.nii' % beta_directory))
beta_aff = beta_sample.affine

# subject_id - get from queue script
subject_id = sys.argv[1]

# Mask whole brain  maps with striatal mask
whole_brain = np.load('/%(1)s/voxel_probabilities_%(2)s.npy' % {"1": voxel_probability_maps_dir, "2": subject_id})
striatum_only = np.zeros((img_dim+[num_networks]))
for network in range(num_networks):
    striatum_only[:,:,:,network] = striatal_mask*whole_brain[:,:,:,network]
striatum_only=np.where(striatum_only==0, np.nan, striatum_only)

# save as numpy
np.save('%(1)s/striatum_only_%(2)s.npy' % {"1": indiv_striatum_proability_maps_np_dir, "2": subject_id}, striatum_only)

#save as nifti
striatum_only_nii = nib.Nifti1Image(striatum_only, beta_aff)
nib.save(striatum_only_nii, '%(1)s/striatum_only_%(2)s.nii' % {"1": indiv_striatum_proability_maps_nii_dir, "2": subject_id})


# group = np.load('/Users/robmcc/Documents/academic/DOPA_symptoms/results/patient_striatum_dd50.npy')
# striatal_mask_temp = nib.load('/Users/robmcc/Documents/academic/DOPA_symptoms/results/arestingstate.nii', mmap=False).get_data() 
# beta_sample = nib.load('/Users/robmcc/Documents/academic/DOPA_symptoms/results/arestingstate.nii', mmap=False)
# beta_aff = beta_sample.affine

# group.shape
# striatum_only_nii = nib.Nifti1Image(group, beta_aff)
# nib.save(striatum_only_nii, '/Users/robmcc/Documents/academic/DOPA_symptoms/results/striatum_groupmap.nii' )

