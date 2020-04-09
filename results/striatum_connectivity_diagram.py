import numpy as np
from configparser import ConfigParser
import glob
from json import loads
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nibabel as nib
# from skimage.transform import resize
from scipy.misc import imresize
import ast

parser = ConfigParser()
#  Amend parser.read statement and amend project_details.txt file depending on where you are running
parser.read('/src/project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
group_map_loc = ('results/group_images/fpn_diagram.npy')
num_networks = int(parser.get('project_details', 'num_networks'))
img_dim = (loads(parser.get("project_details", "image_dimension")))
img_dim = [int(i)for i in img_dim]
images_directory = '%(1)s/results/figures/group_striatum_maps/' % {"1": main_directory}

# Load the patient maps
group_map = np.load(group_map_loc)

#must use nansum otherwise get incorrect denominator as a number fo voxels coded as nan
DMN = np.nansum(group_map[:,:,:,:,0], axis=0)/group_map.shape[0]
SMh = np.nansum(group_map[:,:,:,:,1], axis=0)/group_map.shape[0]
SMm = np.nansum(group_map[:,:,:,:,2], axis=0)/group_map.shape[0]
VIS = np.nansum(group_map[:,:,:,:,3], axis=0)/group_map.shape[0]
FPN = np.nansum(group_map[:,:,:,:,4], axis=0)/group_map.shape[0]
CPN = np.nansum(group_map[:,:,:,:,5], axis=0)/group_map.shape[0]
RST = np.nansum(group_map[:,:,:,:,6], axis=0)/group_map.shape[0]
CON = np.nansum(group_map[:,:,:,:,7], axis=0)/group_map.shape[0]
DAT = np.nansum(group_map[:,:,:,:,8], axis=0)/group_map.shape[0]
AUD = np.nansum(group_map[:,:,:,:,9], axis=0)/group_map.shape[0]
VAT = np.nansum(group_map[:,:,:,:,10], axis=0)/group_map.shape[0]
SAL = np.nansum(group_map[:,:,:,:,11], axis=0)/group_map.shape[0]
Unass = np.nansum(group_map[:,:,:,:,12], axis=0)/group_map.shape[0]


# make group averaged map for PET extraction
sum_acrosspartic = np.nansum(group_map, axis=0)
sum_acrossdim = np.nansum(sum_acrosspartic, axis=3)
b=np.stack([sum_acrossdim]*13, axis=-1)
normalised_map = sum_acrosspartic/b
np.save('%(1)s/results/group_images/patient_normalised.npy' % {"1": main_directory},
        normalised_map)

#colormaps
DMN_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (0.13, 0.53, 1)])
FPN_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (0.2, 0.5, 0)])
AUD_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (0.38, 1, 0.71)])
DAT_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (0.84, 0.15, 0.71)])
SMh_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (1, 0.53, 0.22)])
CON_cmap = LinearSegmentedColormap.from_list('cmapname', [(1, 1, 1),(1, 1, 1), (0.91, 0.89 , 0.15)])

#network_list = ["DMN", "SMh", "SMm", "VIS", "FPN", "CPN", "RST", "CON", "DAT", "AUD", "VAT", "SAL", "Unass" ]
network_list = ["DMN", "AUD", "DAT", "SMh", "CON"]
a=[0,2,4,6,8,10,18,21,24,27,30,33, 36];
b=[x+2 for x in a]

#amend these functions depending on colors thresholding dersired. For colours use: cmap = eval("%s_cmap" % input_array_name) 
# for gray cmap=greys. Need to also change thresholding
def striatum_plot_z(input_array_name):
    input_array =eval(input_array_name)
    input_array[input_array==0]='nan' #remove zeros for normalisation
    input_array=(input_array-np.nanmean(input_array))/np.nanstd(input_array) #normalise 
    # input_array= np.nan_to_num(input_array) #remove Nans for interpolation
    # input_array[input_array > 0] = 0.6
    input_array=input_array-np.nanmin(input_array)
    threshold = np.nanpercentile(input_array, 75)
    # input_array[input_array < threshold] = 0
    input_array[input_array==0]='nan'
    # print(np.nansum(input_array>threshold))
    sns.heatmap(np.flip(np.rot90(imresize(input_array[25:65,50:80,35], 200, mode='F')),axis=1), cmap = eval("%s_cmap" % input_array_name)  , center = 0,  xticklabels = False, yticklabels = False,vmin=0, vmax=np.nanmax(input_array), cbar=False)
    #plt.ylabel(input_array_name)
    # cmap = eval("%s_cmap" % input_array_name) 

def striatum_plot_y(input_array_name):
    input_array =eval(input_array_name)
    input_array[input_array==0]='nan' #normalise
    input_array=(input_array-np.nanmean(input_array))/np.nanstd(input_array) #normalise 
    # input_array= np.nan_to_num(input_array) #replace nans otherwise interpolation fails    thresh = np.nanmean(input_array)  
    input_array=input_array-np.nanmin(input_array)
    threshold = np.nanpercentile(input_array, 75)
    # input_array[input_array < threshold] = 0 #threshold
    # input_array[input_array > 0] = 0.5
    final_image = np.flip(np.rot90(imresize(input_array[25:65,67,28:50], 200, mode='F')), axis=1)
    # print(np.nansum(final_image>threshold))
    final_image[final_image==0]='nan' #replace zeros with nanas so background white
    sns.heatmap(final_image, cmap =eval("%s_cmap" % input_array_name), center = 0, xticklabels = False, yticklabels = False, vmin=0, vmax=np.nanmax(input_array),cbar=False)

np.nanmax(DMN)

plt.rcParams['figure.figsize'] = [14, 5]
grid = plt.GridSpec(12, 10, wspace=0.1, hspace=0.2)

for i in range (len(network_list)):
    plt.subplot(grid[a[i]:b[i],0:4]); striatum_plot_z(network_list[i])
    plt.subplot(grid[a[i]:b[i],5:10]); striatum_plot_y(network_list[i])
    plt.savefig('results/group_images/fpn_diagram.png' , dpi=400)

# response to review re fractional wieghts
for i in range (len(network_list)):
    test=eval(network_list[i])
    test[test==0]='nan'
    print(np.nanmean(test))

a= np.nanpercentile(input_array, 10)
np.sum(input_array>a)
input_array.shape

input_array_name=network_list[i]
input_array =eval(network_list[i])
input_array[input_array==0]='nan' #remove zeros for normalisation
input_array=(input_array-np.nanmean(input_array))/np.nanstd(input_array) #normalise 
threshold = np.nanmean(input_array)

input_array=input_array-np.nanmin(input_array)
input_array[input_array==0]='nan'
sns.heatmap(np.flip(np.rot90(imresize(input_array[25:65,50:80,35], 200, mode='F')),axis=1), cmap = eval("%s_cmap" % input_array_name)  , center = 0,  xticklabels = False, yticklabels = False,vmin=0, vmax=np.nanmax(input_array), cbar=False)
np.nanmax(input_array)
