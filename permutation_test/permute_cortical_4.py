import pandas as pd
import numpy as np
import connectome_builder as cb
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
import os
from collections import defaultdict

# True values 
baseline_data = pickle.load(open("/results/response_permutations/scripted_bl.pkl", "rb"))
baseline_symptom_data =  baseline_data.loc[:, ['marder_dep', 'marder_disorg',  'marder_exc', 'marder_neg', 'marder_pos']]
baseline_subdiv_data = baseline_data.loc[:,['AUD', 'CON', 'DAT', 'DMN', 'SMh', 'VIS', 'wstr', 'ast', 'lst', 'smst']].dropna(axis=1)
[true_corr_bl, true_p_bl] = cb.cor_matrix(baseline_symptom_data.join(baseline_subdiv_data), 'pearson')
true_bl={}
for network in ['AUD','DMN','SMh','CON','DAT']:
    for marder_domain in ['pos', 'neg', 'disorg', 'exc', 'dep']:
        true_bl[f'{marder_domain}_{network}_z']= true_corr_bl.loc[(f'marder_{marder_domain}'), network]

# Cortical permutations
number_permutations=10000
cortical_bl_r = pickle.load(open("/results/response_permutations/baseline_corticalpermuted_scripted.pkl", "rb"))
permute_cortical_bl = defaultdict(list)
for i in range(number_permutations):    
    shuffled_corr_bl = cortical_bl_r[i]    
    for network in ['Net 1','Net 2', 'Net 8','Net 9', 'Net 10']:
        for marder_domain in ['pos', 'neg', 'disorg', 'exc', 'dep']:
            z = shuffled_corr_bl.loc[(f'marder_{marder_domain}'), network]
            permute_cortical_bl[f'{marder_domain}_{network}_z'].append(z)

# Get keys matching
old_keys = dict(permute_cortical_bl).keys()
swap_dict = {'AUD_':'Net 10_', 'DMN_': 'Net 1_', 'SMh_':'Net 2_',  'CON_':'Net 8_', 'DAT_':'Net 9_'}
reversed_dict = dict(map(reversed, swap_dict.items()))
old_nets = swap_dict.values()

for old_key in old_keys:
    for old_net in old_nets:
        if old_net in old_key:
            new_net = reversed_dict[old_net]
            amended_key = old_key.replace(old_net, new_net)
            permute_cortical_bl[amended_key] = permute_cortical_bl[old_key]
            del permute_cortical_bl[old_key]


# Results
def perm_test(true, null, variable):
        return np.sum(true[variable]<=np.asarray(null[variable]))/len(null[variable])

perm_result = 'permute_cortical_bl'
results_dict = {}
for test_variable in true_bl.keys():
        p = perm_test(true_bl, eval(perm_result), test_variable)
        results_dict.update({test_variable: p})

fdrcorrection(list(results_dict.values()))


# 3 Figures
sns.set_style("ticks")

def perm_graphs(plot_variables, colors, permuted_data, true_data, labels):
    for i, plot_variable in enumerate(plot_variables):
        sns.kdeplot(permuted_data[plot_variable], color=colors[i], linewidth=2)
        plt.vlines(x=true_data[plot_variable], ymin=0, ymax=2.0, color=colors[i], linewidth=4.0)
    plt.title(labels[0]), plt.xlabel(labels[1]), plt.ylabel(labels[2])
    plt.xlim(-1.25, 0.75)


def graph4grid(z_variables, zdiff_variables, colors, permute_bl, permute_ch, true_bl, true_ch, title):
    plt.rcParams['figure.figsize'] = [10, 10]
    sns.set(rc={'axes.facecolor':'#F8F8F8'})
    grid = plt.GridSpec(5, 5, wspace=0.0, hspace=5.0)
    
    fig, axes = plt.subplots(4, 4)
    plt.suptitle(title, fontsize= 12)

    labels = ['Baseline Symptoms', 'Z', 'Frequency']
    plt.subplot(221)
    perm_graphs(z_variables, colors, permute_bl, true_bl, labels)

    labels = ['Symptom Change', 'Z', 'Frequency']
    plt.subplot(223)
    perm_graphs(z_variables, colors, permute_ch, true_ch, labels)

    labels = ['Baseline Symptoms', 'Z Difference', 'Frequency']
    plt.subplot(222)
    perm_graphs(zdiff_variables, colors, permute_bl, true_bl, labels)

    labels = ['Symptom Change', 'Z Difference', 'Frequency']
    plt.subplot(224)
    perm_graphs(zdiff_variables, colors, permute_ch, true_ch, labels)

    plt.tight_layout(pad=3.5, w_pad=0.5, h_pad=2.0)


# Individual
colors = ['#60ffb6ff', '#ff8838ff', '#e8e227ff', '#2287ffff']
z_variables = ['aud_z', 'smh_z', 'con_z', 'custom_z',]
zdiff_variables = ['aud_diff', 'smh_diff', 'con_diff', 'custom_diff']
graph4grid(z_variables, zdiff_variables, colors, permute_indiv_bl, permute_indiv_ch, true_bl, true_ch, 'Participants Permuted')
plt.savefig('/results/figures/graphs/permutation_results/individual_perm.png', dpi=400)


# Cortical
colors = ['#60ffb6ff', '#ff8838ff', '#e8e227ff', '#2287ffff']
z_variables = ['aud_z', 'smh_z', 'con_z', 'custom_z']
zdiff_variables = ['aud_diff', 'smh_diff', 'con_diff', 'custom_diff']


graph4grid(z_variables, zdiff_variables, colors, permute_cortical_bl, permute_cortical_ch, true_bl, true_ch, 'ROIs Permuted')
plt.savefig('/figures/graphs/permutation_results/cortical_perm_woline.png', dpi=400)
