# Load modules
from configparser import ConfigParser
import numpy as np
import pandas as pd
import ast
import connectome_builder as cb
import bct
import copy
from importlib import reload
import pickle

# Load data
parser = ConfigParser()
parser.read('project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
clinical_data = pd.read_csv(main_directory+'clinical.csv')
clinical_data = clinical_data.loc[clinical_data['include'] == 1]
idx = np.where(clinical_data['include'].dropna())
connectomes = np.load(main_directory+'results/cortical_connectomes/connectomes.npy')
connectomes_orig = copy.deepcopy(connectomes)
connectomes = np.squeeze(connectomes_orig[:, :, idx])
apriori_communities = parser._sections['original_gordon_node_ids']
networks = parser._sections['network_key']

# Make distance array (fill empty lower triangle)
distance_array = np.load(main_directory+'results/distance_array.npy')
i_lower = np.tril_indices(distance_array.shape[0], -1)
distance_array[i_lower] = distance_array.T[i_lower]
distance_mask = distance_array > 10

# Generate matrix for clustering: normalisation and distance threshold
W_norm = np.zeros(connectomes.shape)
n = connectomes.shape[0]
for i in range(connectomes.shape[2]):
    individual_graph = connectomes[:, :, i]
    std_dev = np.std(np.tril(individual_graph))
    W_norm[:, :, i] = (individual_graph-(np.sum(np.sum(individual_graph))/(n**2-n)))/std_dev
W_mn = np.mean(W_norm, 2)
np.fill_diagonal(W_mn, 0)
thresholded_matrix = bct.threshold_proportional(W_mn, 0.05)
thresholded_matrix = np.where(thresholded_matrix > 0, 1, 0)

LouvainMethod = 'modularity'
gamma=1.4
consensusMatrixThreshold = 0.5
numberPartition = 50
distance_thresholded_matrix = thresholded_matrix*distance_mask
seeds = range(numberPartition) 
[ignore1, ignore2, community_allocation] = cb.consensus_clustering_louvain(distance_thresholded_matrix,
                                                                        numberPartition, consensusMatrixThreshold,
                                                                        LouvainMethod, gamma, seeds)


# Look at how to label these networks
columns = list(apriori_communities.keys())
columns.insert(0, "assigned community")
community_copy = pd.DataFrame(index=list(range(1,(len(community_allocation)+1))), columns=columns)

for community in range(1, (len(community_allocation)+1)):
    nodes = np.squeeze(community_allocation[community])
    nodes = nodes+1  #adjust as these node labels start at zero
    for a_priori_community in apriori_communities.keys():
        apriori_nodes = np.array(ast.literal_eval(apriori_communities[a_priori_community]))
        overlap = len(np.intersect1d(nodes, apriori_nodes))
        community_copy.loc[community, a_priori_community] = overlap
    # label the top scoring overlap
    top_network_idx = np.nanargmax(community_copy.loc[community, :])
    community_copy.loc[community, 'assigned community'] = columns[top_network_idx]

for network in networks:
    node_ids=[]
    idx=community_copy.index[community_copy["assigned community"]==network]
    if idx.shape[0]!=0:
        node_ids = np.concatenate([community_allocation[k][0][0]+1 for k in idx])
        if remove_none==1:
            node_ids = np.asarray(replace_indices(node_ids, none_list))
        print(network, np.array2string(node_ids, separator=','))

# dice dmn-fpn (response to reviewer)
import json
node_assignment=parser._sections['node_ids']
gordon_orig=parser._sections['original_gordon_node_ids']

dmn_me=json.loads(node_assignment['default'])
dmn_gordon=json.loads(gordon_orig['default'])
fpn_gordon=json.loads(gordon_orig['frontoparietal'])

2*len(np.intersect1d(fpn_gordon, dmn_me))/(len(fpn_gordon)+len(dmn_me))

2*s/(len(fpn_gordon)+len(dmn_me))

