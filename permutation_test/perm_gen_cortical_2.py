import pandas as pd
import numpy as np
import connectome_builder as cb
import pickle
import seaborn as sns
import os
import pickle
import sys

# Permuting cortical rois
node_assignment='node_ids'
number_permutations = 10000
number_networks = 13
number_subjects = 29
permuted_data = np.zeros([number_permutations+1, number_networks, number_subjects+1])
clinical_data = pd.read_csv('clinical.csv')
clinical_data = clinical_data.loc[clinical_data['include']==1]
baseline_data = clinical_data.loc[clinical_data['patient']==1]
pvalue_list = []
rvalue_list =[]

#add in marder factors
baseline_data.insert(1, 'marder_pos', baseline_data.loc[:, ['p1','p3', 'p5','p6','n7', 'g1', 'g9', 'g12']].sum(axis=1, skipna=False))
baseline_data.insert(1, 'marder_neg', baseline_data.loc[:, ['n1','n2', 'n3', 'n4', 'n6', 'g7', 'g16']].sum(axis=1, skipna=False))
baseline_data.insert(1, 'marder_disorg', baseline_data.loc[:, ['p2','n5', 'g5', 'g11', 'g13', 'g15', 'g10']].sum(axis=1, skipna=False))
baseline_data.insert(1, 'marder_exc', baseline_data.loc[:, ['p4','p7', 'g8', 'g14']].sum(axis=1, skipna=False))
baseline_data.insert(1, 'marder_dep', baseline_data.loc[:, ['g2','g3', 'g4', 'g6']].sum(axis=1, skipna=False))

perm_string=sys.argv[1]
perm_list = perm_string.split(",")

df_index=0
for i in baseline_data['subject_id']:
    subject_id = '%03d' %i
    permuted_data[:,:,df_index] =np.load(f'results/null_dist/cortical_perm/{node_assignment}/network_kis_{subject_id}.npy')
    df_index = df_index+1

example_ki_perm = permuted_data[1,:,:].T
column_names = ["Net %d" % (i + 1) for i in range(example_ki_perm.shape[1])]
symptoms = clinical_data.loc[:,'p1':'g16'].columns
symptoms = symptoms.append(pd.Index(['marder_pos', 'marder_neg', 'marder_disorg', 'marder_exc', 'marder_dep']))
select_columns = np.union1d(symptoms,["Net 1", "Net 2", "Net 8", "Net 9", "Net 10"])

for i in range(int(perm_list[0]),int(perm_list[1])):
    ki_perm = permuted_data[i,:,:].T
    ki_perm_data = pd.DataFrame(data = ki_perm[0:29,:], columns = column_names, index = baseline_data['subject_id'].values).dropna(axis=1)
    ki_perm_data['subject_id'] = baseline_data['subject_id'].values
    perm_data = baseline_data.join(ki_perm_data.set_index('subject_id'), on='subject_id')
    required_data = perm_data[select_columns].dropna(axis=0)
    [perm_direct_corr_r, perm_direct_corr_p] = cb.cor_matrix(required_data, 'pearson')
    pvalue_list.append(perm_direct_corr_p)
    rvalue_list.append(perm_direct_corr_r)

results = [rvalue_list, pvalue_list]
pickle.dump(results, open(f'results/response_permutations/baseline_corticalpermuted_{node_assignment}{str(perm_list[0])}.pkl', "wb"))
