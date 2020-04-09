from configparser import ConfigParser
import nibabel as nib
import numpy as np
import sys
import os

parser = ConfigParser()
parser.read('/project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
beta_directory = parser.get('project_details', 'beta_directory')
node_selection=sys.argv[1]
print(sys.executable.split('/')[-3])
# change this as needed
striatum_probmaps_dir = f'{ main_directory}results/indiv_striatum_probability_maps/{node_selection}/numpy' 
pet_maps_dir = ('%sdata/ki_maps' % main_directory)
num_networks = int(parser.get('project_details', 'num_networks'))

# change this too
pet_csv_path = f'{main_directory}results/pet_network_kis/{node_selection}.csv'
subject_list_file = parser.get('project_details', 'subject_list_path')

with open(subject_list_file, 'r') as f:
    for subject_id_init in f:
        subject_id = subject_id_init[0:3]  # need to slice off so '001/n'-> '001'

        # Load PET beta_map
        pet_filename = os.listdir('%(1)s/%(2)s/' % {"1": pet_maps_dir, "2": subject_id})
        pet_map_nii = nib.load('%(1)s/%(2)s/%(3)s' % {"1": pet_maps_dir, "2": subject_id, "3": pet_filename[0]})
        #pet_map_nii = nib.load('%(1)s/z_pet_%(2)s.nii' % {"1": pet_maps_dir, "2": subject_id})
        pet_map = np.array(pet_map_nii.get_data())

        # Load striatal voxel_assignments
        striatal_map = np.load('%(1)s/striatum_only_%(2)s.npy' % {"1": striatum_probmaps_dir, "2": subject_id})


        # Mean ki for each networks
        # Array to store mean ki for each network
        network_kis = np.zeros([2, num_networks])
        network_kis[0, :] = range(1, num_networks+1)

        for network in range(1, num_networks+1):
            # Mask pet map for each network and weight by probabilities(nb - TRUE means the value is masked)
            probability_sum = np.nansum(striatal_map[:, :, :, network-1])
            weighted_ki = np.multiply(striatal_map[:, :, :, network-1], pet_map)
            network_kis[1, network-1] = np.nansum(weighted_ki)/probability_sum

        # Open csv and append row,  first column is subject ID
        # Data to write
        save_to_csv = np.column_stack([[int(subject_id)], [network_kis[1, :]]])
        print(save_to_csv)
        # Open the file in append mode
        pet_csv = open(pet_csv_path, 'ab')
        np.savetxt(pet_csv, save_to_csv, delimiter=",")
        pet_csv.close()

        if 'str' in subject_id_init:
            break

print('finished')