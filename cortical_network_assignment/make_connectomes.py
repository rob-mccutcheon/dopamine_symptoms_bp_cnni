
from configparser import ConfigParser
import numpy as np
from connectome_builder import create_connectomes

# Load directory and file ids
# If running locally the path: '/Users/robmcc/mnt/droplet/home/k1201869/DOPA_symptoms/src/project_details.txt'
# Also amend project_details.txt file depending on where you are running
# If running on the NaN: '/home/k1201869/DOPA_symptoms/src/project_details.txt'
parser = ConfigParser()
parser.read('/home/k1201869/DOPA_symptoms/src/project_details.txt')
#parser.read('/Users/robmcc/mnt/droplet/home/k1201869/DOPA_symptoms/src/project_details.txt')

main_directory = parser.get('project_details', 'main_directory')
connectome = parser.get('project_details', 'connectome')
connectomes_directory = (main_directory+'results/cortical_connectomes/connectomes.npy')
num_rois = 333
num_partic = 52

connectomes = create_connectomes(connectome, num_rois, num_partic)
np.save(connectomes_directory, connectomes)
