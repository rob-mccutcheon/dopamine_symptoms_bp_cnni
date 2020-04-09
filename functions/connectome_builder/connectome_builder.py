from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
import bct  # Edited the louvain function as  crashing commented out lines 110-111 of modularity.py and corrected bugs
from collections import defaultdict
import nibabel as nib
from scipy import stats, linalg


def create_connectomes(conn_ROI_path, num_rois, num_partic):
    rawdata = loadmat(conn_ROI_path)
    ROI = rawdata['ROI']
    values = range(0, num_rois)
    init_array = np.empty((1, num_rois), dtype=object)
    connectomes = np.empty([num_rois, num_rois, num_partic])

    for val in values:
        init_array[0, val] = (ROI[0, val][1])[:, val:num_rois]

    data = np.hstack(init_array.flatten())
    nans_removed = data[:, ~np.all(np.isnan(data), axis=0)]

    for i in range(num_partic):
        a = nans_removed[i, ]
        # create top triangle of matrix
        b = np.triu(np.ones(num_rois), 1)
        b[b.astype(bool)] = a
        # Fill the 3d array
        connectomes[:, :, i] = b + b.transpose()

    return (connectomes)


def consensus_clustering_louvain(inputMatrix, numberPartitions, consensusMatrixThreshold, LouvainMethod, gamma, seeds):
    # Function to implement consensus clustering as per Lancichinetti & Forunato et al 2012
    # using the Louvain algorithm as implemented by BCT
    #
    # Inputs:
    #   inputMatrix                 symmetrical weighted undirected adjacency matrix to be partiitioned
    #   numberIterations            number of times the algorithm is run to  generate the consensus matrix on each run
    #   consensusMatrixThreshold    threshold below which consensus matrix  entries are st to zero, (0 1]
    #   LouvainMethod               string of Louvain method: 'Modularity' (if no negative weights in the inputMatrix,
    #                               or 'negative_sym' / 'negative_asym' if negative weights)
    #   gamma                       resolution parameter of Louvain
    #   eeds                       array of integers equal in length to the number of partitions
    #
    # Outputs:
    #   finalPartition              final community allocaiton of each node
    #   iterations                  how many iterations to reach consensus
    #   communityAssignment         final community assignment

    consensus = False
    iterations = 0

    while not consensus:
        D = np.zeros((inputMatrix.shape[0], inputMatrix.shape[1], numberPartitions))  # consensus matrix
        # generate consensus matrix
        for partition in range(numberPartitions):
            [community_allocation, ignore_Q] = bct.community_louvain(inputMatrix, gamma=gamma, ci=None, B=LouvainMethod, seed=seeds[partition])

            for row in range(D.shape[0]):
                for col in range(D.shape[1]):
                    D[row, col, partition] = (community_allocation[row] == community_allocation[col])

        D = np.mean(D, 2)  # consensus matrix...is it equal or do we need to keep going?
        iterations = iterations + 1  # keep track

        if np.unique(D).shape[0] < 3:  # only true if all parition matrices equal (so that their mean is either 0 or 1)
            consensus = True
            finalPartition = D
            communityAssignment = defaultdict(list)
            for community in range(1, np.unique(community_allocation).shape[0]+1):
                communityAssignment[community].append(np.where(community_allocation == community))

        else:
            D = np.where(D < consensusMatrixThreshold, 0, D)
            inputMatrix = D

    return(finalPartition, iterations, communityAssignment)


def consensus_clustering_louvain2(inputMatrix, numberPartitions, consensusMatrixThreshold, LouvainMethod, gamma, seeds):
    # JUST AMENDED SO THAT WE GET Q OUT AT THE END
    # Function to implement consensus clustering as per Lancichinetti & Forunato et al 2012
    # using the Louvain algorithm as implemented by BCT
    #
    # Inputs:
    #   inputMatrix                 symmetrical weighted undirected adjacency matrix to be partiitioned
    #   numberIterations            number of times the algorithm is run to  generate the consensus matrix on each run
    #   consensusMatrixThreshold    threshold below which consensus matrix  entries are st to zero, (0 1]
    #   LouvainMethod               string of Louvain method: 'Modularity' (if no negative weights in the inputMatrix,
    #                               or 'negative_sym' / 'negative_asym' if negative weights)
    #   gamma                       resolution parameter of Louvain
    #   seeds                       'None' or an integer
     #
    # Outputs:
    #   finalPartition              final community allocaiton of each node
    #   iterations                  how many iterations to reach consensus
    #   communityAssignment         final community assignment

    consensus = False
    iterations = 0

    while not consensus:
        D = np.zeros((inputMatrix.shape[0], inputMatrix.shape[1], numberPartitions))  # consensus matrix
        # generate consensus matrix
        for partition in range(numberPartitions):
            [community_allocation, Q] = bct.community_louvain(inputMatrix, gamma=gamma, ci=None, B=LouvainMethod, seed=seeds[partition])

            for row in range(D.shape[0]):
                for col in range(D.shape[1]):
                    D[row, col, partition] = (community_allocation[row] == community_allocation[col])

        D = np.mean(D, 2)  # consensus matrix...is it equal or do we need to keep going?
        iterations = iterations + 1  # keep track

        if np.unique(D).shape[0] < 3:  # only true if all parition matrices equal (so that their mean is either 0 or 1)
            consensus = True
            finalPartition = D
            communityAssignment = defaultdict(list)
            for community in range(1, np.unique(community_allocation).shape[0]+1):
                communityAssignment[community].append(np.where(community_allocation == community))

        else:
            D = np.where(D < consensusMatrixThreshold, 0, D)
            inputMatrix = D

    return(Q, finalPartition, iterations, communityAssignment)



def atlas2rois(atlas_file, roi_directory):
    #turns an atlas file into indivdual rois and saves them
    atlas = nib.load(atlas_file, mmap=False)
    affine = atlas.affine
    for i in range(1, int(np.max(atlas.get_data()))+1):
        roi = np.where(atlas.get_data()==i,1,0)
        roi_nii = nib.Nifti1Image(roi, affine)
        roi_nii.header.set_data_dtype('float32')
        nib.save(roi_nii, ('%(1)sroi_%(2)s.nii' % {"1":roi_directory, "2": i}))



def cor_matrix(dataframe, method):
    coeffmat = pd.DataFrame(index=dataframe.columns, columns=dataframe.columns)
    pvalmat = pd.DataFrame(index=dataframe.columns, columns=dataframe.columns)
    for i in range(dataframe.shape[1]):
        for j in range(dataframe.shape[1]):
            x = np.array(dataframe[dataframe.columns[i]])
            y = np.array(dataframe[dataframe.columns[j]])
            bad = ~np.logical_or(np.isnan(x), np.isnan(y))
            if method == 'spearman':
                corrtest = spearmanr(np.compress(bad,x), np.compress(bad,y))
            if method == 'pearson':
                corrtest = pearsonr(np.compress(bad,x), np.compress(bad,y))
            coeffmat.iloc[i,j] = corrtest[0]
            pvalmat.iloc[i,j] = corrtest[1]
    #This is to convert to float type otherwise can cause problems when e.g. plotting
    coeffmat=coeffmat.apply(pd.to_numeric, errors='ignore')
    pvalmat=pvalmat.apply(pd.to_numeric, errors='ignore')
    return (coeffmat, pvalmat)

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    P_pval = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = stats.pearsonr(res_i, res_j)
            P_corr[i, j] = corr[0]
            P_corr[j, i] = corr[0]
            P_pval[i, j] = corr[1]
            P_pval[j, i] = corr[1]

    return (P_corr, P_pval)
