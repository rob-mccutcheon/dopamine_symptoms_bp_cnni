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
pet_csv_path = ('results/pet_network_kis/scripted.csv')
pet_data = pd.read_csv(pet_csv_path, names=['subject_id', 'DMN', 'SMh', 'SMm', 'VIS', 'FPN', 'CPN', 'RST', 'CON', 'DAT', 'AUD', 'VAT','SAL',' None'], index_col=None)
clinical_data = pd.read_csv('/clinical.csv')
clinical_data = clinical_data.loc[clinical_data['include']==1]
dopa_data = clinical_data.join(pet_data.set_index('subject_id'), on = 'subject_id')
dopa_data.insert(1, 'marder_pos', dopa_data.loc[:, ['p1','p3', 'p5','p6','n7', 'g1', 'g9', 'g12']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_neg', dopa_data.loc[:, ['n1','n2', 'n3', 'n4', 'n6', 'g7', 'g16']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_disorg', dopa_data.loc[:, ['p2','n5', 'g5', 'g11', 'g13', 'g15', 'g10']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_exc', dopa_data.loc[:, ['p4','p7', 'g8', 'g14']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_dep', dopa_data.loc[:, ['g2','g3', 'g4', 'g6']].sum(axis=1, skipna=False))
baseline_data = dopa_data.loc[dopa_data['patient']==1]
baseline_data.to_pickle('response_permutations/scripted_bl.pkl')
baseline_symptom_data =  baseline_data.loc[:, ['marder_dep', 'marder_disorg',  'marder_exc', 'marder_neg', 'marder_pos']]
baseline_subdiv_data = baseline_data.loc[:,['AUD', 'CON', 'DAT', 'DMN', 'SMh', 'VIS', 'wstr', 'ast', 'lst', 'smst']].dropna(axis=1)
[bl_corr, true_p_bl] = cb.cor_matrix(baseline_symptom_data.join(baseline_subdiv_data), 'pearson')

plt.rcParams['figure.figsize'] = [10, 8]
subdivs=['DMN','SMh', 'CON','DAT', 'AUD', 'lst', 'ast', 'smst']
sns.heatmap(bl_corr.loc[subdivs,subdivs[::-1]], cmap="Reds", vmin = 0, vmax=1, center=0.5, cbar=True)
plt.axhline(y=5)
plt.axvline(x=3)
plt.savefig('/results/figures/graphs/ki_correl.png', dpi=400)

plt.rcParams['figure.figsize'] = [10, 7]
p=pd.read_csv('results/kis_correlation_signif.csv')
z_data=pd.read_csv('results/kis_correlation_z.csv')
index = ['LST-AST', 'LST-SMST', 'AST-SMST']
columns = ['id','DMN-AUD', 'DMN-DAT', 'DMN-SMN', 'DMN-CON','AUD-DAT', 'AUD-SMN', 'AUD-CON','DAT-SMN', 'DAT-CON', 'SMN-CON']
z= pd.DataFrame(index=index,columns =columns, data=z_data.values)
sns.heatmap(p.iloc[:,1:]<0.05, vmin=0, vmax=0.1)
plt.rcParams['figure.figsize'] = [10, 7]
sns.heatmap(z.iloc[:,1:], vmin=-6, vmax= 0, cmap='Reds_r') 
plt.yticks(rotation=0) 
x=np.where(p<0.05)[1]
y=np.where(p<0.05)[0]
for i in range(len(x)):
    plt.text(x[i]-0.6,y[i]+0.6,'*', fontsize=26, color = 'white')
plt.savefig('/results/figures/graphs/correl_comparison.png', dpi=400)