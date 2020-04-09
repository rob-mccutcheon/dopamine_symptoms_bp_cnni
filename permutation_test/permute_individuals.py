import pandas as pd
import numpy as np
import connectome_builder as cb
import pickle
import seaborn as sns
import os
from collections import defaultdict
from statsmodels.stats.multitest import fdrcorrection
from matplotlib import pyplot as plt

# Load Data
pet_csv_path = (results/pet_network_kis/node_ids.csv')
pet_data = pd.read_csv(pet_csv_path, names=['subject_id', 'DMN', 'SMh', 'SMm', 'VIS', 'FPN', 'CPN', 'RST', 'CON', 'DAT', 'AUD', 'VAT','SAL',' None'], index_col=None)
clinical_data = pd.read_csv('/results/copies_of_local/clinical.csv')
clinical_data = clinical_data.loc[clinical_data['include']==1]
dopa_data = clinical_data.join(pet_data.set_index('subject_id'), on = 'subject_id')
dopa_data.insert(1, 'marder_pos', dopa_data.loc[:, ['p1','p3', 'p5','p6','n7', 'g1', 'g9', 'g12']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_neg', dopa_data.loc[:, ['n1','n2', 'n3', 'n4', 'n6', 'g7', 'g16']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_disorg', dopa_data.loc[:, ['p2','n5', 'g5', 'g11', 'g13', 'g15', 'g10']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_exc', dopa_data.loc[:, ['p4','p7', 'g8', 'g14']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_dep', dopa_data.loc[:, ['g2','g3', 'g4', 'g6']].sum(axis=1, skipna=False))
# dopa_data.insert(1, 'classic_pos', dopa_data.loc[:, ['p1','p2','p3', 'p4','p5','p6','p7']].sum(axis=1, skipna=False))
baseline_data = dopa_data.loc[dopa_data['patient']==1]
baseline_data.to_pickle('/results/response_permutations/scripted_bl.pkl')
baseline_symptom_data =  baseline_data.loc[:, ['marder_dep', 'marder_disorg',  'marder_exc', 'marder_neg', 'marder_pos']]
baseline_subdiv_data = baseline_data.loc[:,['AUD', 'CON', 'DAT', 'DMN', 'SMh', 'VIS', 'wstr', 'ast', 'lst', 'smst']].dropna(axis=1)
[true_corr_bl, true_p_bl] = cb.cor_matrix(baseline_symptom_data.join(baseline_subdiv_data), 'pearson')


# Permute individuals
num_perm = 10000
array_size = baseline_symptom_data.shape[1] + baseline_subdiv_data.shape[1]
permute_indiv=np.empty([array_size, array_size, num_perm])
corr_diff=np.empty([array_size, array_size, num_perm])
shuffled_baseline_data =baseline_symptom_data.copy(deep=True)

for i in range(num_perm):
    np.random.seed(i) 
    shuffled_baseline_data = shuffled_baseline_data.set_index(np.random.permutation(baseline_data.index))
    shuffled_baseline_combined = shuffled_baseline_data.join(baseline_subdiv_data)
    [shuffled_corr_bl, shuffled_p_bl] = cb.cor_matrix(shuffled_baseline_combined, 'pearson')
    permute_indiv[:,:,i] = shuffled_corr_bl >=true_corr_bl
    corr_diff[:,:,i] = shuffled_corr_bl

# Save results
results = [permute_indiv, corr_diff]
pickle.dump(results, open('/results/response_permutations/scripted_martinez.pkl', "wb"))
[permute_indiv, corr_diff] = pickle.load(open('/results/response_permutations/scripted_martinez.pkl', "rb"))

# FDR correction
permuted_ps = np.mean(permute_indiv, 2)
ps = permuted_ps[baseline_symptom_data.shape[1]:, :baseline_symptom_data.shape[1]]
connectivity_ps = ps[0:5]
martinez_ps = ps[5:,]
fdrcorrection(connectivity_ps.flatten())
fdrcorrection(martinez_ps.flatten())



subdivs = ['AUD', 'CON', 'DAT', 'DMN', 'SMh', 'wstr', 'ast', 'lst','smst']
symptoms=['marder_dep', 'marder_disorg', 'marder_exc', 'marder_neg', 'marder_pos']
p_perm=pd.DataFrame(index=subdivs, columns=symptoms, data =ps)
p_perm=p_perm.transpose()
plt.figure(figsize=(10,7))
sns.heatmap(true_corr_bl.loc['marder_dep':'marder_pos', subdivs], cmap="RdBu_r", vmin = -0.1, vmax=0.6, center=0)

x=np.where(p_perm.loc['marder_dep':'marder_pos', subdivs]<0.05)[1]
y=np.where(p_perm.loc['marder_dep':'marder_pos', subdivs]<0.05)[0]
for i in range(len(x)):
    plt.text(x[i]+0.25,y[i]+0.75,'*', fontsize=26, color = 'white')
plt.axvline(x=5)
plt.savefig('/results/figures/ki_symp_heatmap.png', dpi=400)

# Compare correlation coefficents
symptoms = [0,1,2,3,4]
networks = np.arange(5,14)
network_labels = ['AUD', 'CON', 'DAT', 'DMN', 'SMN', 'WST', 'AST', 'LST', 'SMST']
symptom=3
num_perm=10000
for symptom in symptoms:
    # results_corr = np.empty([len(networks), len(networks)])
    results_corr = pd.DataFrame(index=network_labels, columns=network_labels).astype(float)
    results_corr2 = pd.DataFrame(index=network_labels, columns=network_labels).astype(float)
    for network1 in networks:
        for network2 in networks:
            true = abs(true_corr_bl.iloc[symptom,  network1]-true_corr_bl.iloc[symptom,network2])
            true2 = true_corr_bl.iloc[symptom,  network1]-true_corr_bl.iloc[symptom,network2]
            p=np.sum(true <= abs(corr_diff[symptom, network1,:]-corr_diff[symptom,network2,:]))/num_perm
            # p=np.sum(true2 <= (corr_diff[symptom, network1,:]-corr_diff[symptom,network2,:]))/num_perm
            results_corr.iloc[network1-networks[0], network2-networks[0]] = p
            results_corr2.iloc[network1-networks[0], network2-networks[0]] = true2
    plt.figure(figsize=(10,10))
    # sns.heatmap(results_corr2, cmap="Reds_r", vmin = 0, vmax=1, center=0.5, cbar=True)
    sns.heatmap(results_corr2, cmap="RdBu_r", vmin = -0.7, vmax=0.7, center=0.0, cbar=True)
    plt.axhline(y=5)
    plt.axvline(x=5)
    plt.title(f'symptoms_{symptom}')
    x=np.where(results_corr<0.05)[1]
    y=np.where(results_corr<0.05)[0]
    for i in range(len(x)):
        plt.text(x[i]+0.25,y[i]+1,'*', fontsize=26, color = 'white')
    plt.savefig(f'/results/figures/compare_symp_correl/symptom{symptom}_corr.png', dpi=400)
    plt.close()

