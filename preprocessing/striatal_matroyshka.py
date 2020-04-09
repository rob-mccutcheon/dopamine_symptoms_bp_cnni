from configparser import ConfigParser
import nibabel as nib
import numpy as np
import scipy.ndimage

# Load file details
parser = ConfigParser()
parser.read('/project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
beta_directory = parser.get('project_details', 'beta_directory')
roi_directory = parser.get('project_details', 'roi_directory')
striatum_file = parser.get('project_details', 'striatum_file')

# Load affine from a beta file
beta_sample = (nib.load('%sBETA_Subject001_Condition001_Source001.nii' % beta_directory))
beta_aff = beta_sample.affine

# Load striatal mask and binarise
striatal_mask_nii = (roi_directory+striatum_file)
striatal_mask_np = nib.load(striatal_mask_nii, mmap=False).get_data()  # disables 'memmap' special arrays
striatal_mask_np = striatal_mask_np > 0

# Dilate image
enlarged_striatum = scipy.ndimage.binary_dilation(striatal_mask_np, iterations=1)
dilated_striatum = scipy.ndimage.binary_dilation(striatal_mask_np, iterations=3)
striatal_matroyshka = dilated_striatum & ~enlarged_striatum
striatal_matroyshka = striatal_matroyshka.astype(float)
striatal_matroyshka

# Save as numpy
np.save('%(1)sresults/striatal_matroyshka/striatal_matroyshka.npy' % {"1": main_directory}, striatal_matroyshka)

# Save as nifti
striatal_matroyshka_nii = nib.Nifti1Image(striatal_matroyshka, beta_aff)
nib.save(striatal_matroyshka_nii, '%(1)sresults/striatal_matroyshka/striatal_matroyshka.nii' % {"1": main_directory})
