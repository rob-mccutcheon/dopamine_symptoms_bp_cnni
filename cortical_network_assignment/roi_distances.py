import connectome_builder as cb


#Convert atlas to rois
atlas_file = 'ROIS/Gordon/Parcels_MNI_222.nii'
newrois_directory = 'ROIS/Gordon/individual_rois/'
cb.atlas2rois(atlas_file, newrois_directory)

# distances between rois
from nipype.algorithms import metrics
import nibabel as nib
import os
roi_list = os.listdir(newrois_directory)
distance = metrics.Distance()
distance_array = np.zeros([len(roi_list),len(roi_list)])
for i in range (1,len(roi_list)+1):
    for j in range (i,len(roi_list)+1):
        roi1 = ('%(1)sroi_%(2)s.nii' % {"1":newrois_directory, "2": i})
        roi2 = ('%(1)sroi_%(2)s.nii' % {"1":newrois_directory, "2": j})
        distance.inputs.volume1 = roi1
        distance.inputs.volume2 = roi2
        distance.inputs.method = 'eucl_min'
        result=distance.run()
        distance_array[i-1,j-1] = result.outputs.distance