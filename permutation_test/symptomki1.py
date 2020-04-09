# Load Modules
import pandas as pd
import numpy as np
import connectome_builder as cb
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import bct


# Load Data
pet_csv_path = ('/home/k1201869/DOPA_symptoms/results/pet_network_kis/scripted.csv')
pet_data = pd.read_csv(pet_csv_path, names=['subject_id', 'DMN', 'SMh', 'SMm', 'VIS', 'FPN', 'CPN', 'RST', 'CON', 'DAT', 'AUD', 'VAT','SAL',' None'], index_col=None)
clinical_data = pd.read_csv('/home/k1201869/DOPA_symptoms/results/copies_of_local/clinical.csv')
clinical_data = clinical_data.loc[clinical_data['include']==1]
dopa_data = clinical_data.join(pet_data.set_index('subject_id'), on = 'subject_id')
dopa_data.insert(1, 'marder_pos', dopa_data.loc[:, ['p1','p3', 'p5','p6','n7', 'g1', 'g9', 'g12']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_neg', dopa_data.loc[:, ['n1','n2', 'n3', 'n4', 'n6', 'g7', 'g16']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_disorg', dopa_data.loc[:, ['p2','n5', 'g5', 'g11', 'g13', 'g15', 'g10']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_exc', dopa_data.loc[:, ['p4','p7', 'g8', 'g14']].sum(axis=1, skipna=False))
dopa_data.insert(1, 'marder_dep', dopa_data.loc[:, ['g2','g3', 'g4', 'g6']].sum(axis=1, skipna=False))
patient_data = dopa_data.loc[dopa_data['patient']==1]
patient_data.to_pickle('/home/k1201869/DOPA_symptoms/results/response_permutations/scripted.pkl')


pd.options.display.max_seq_items = 2000
print(response_data.columns)

# Baseline correlation matrix 
def make_indices(df, panss1, panss2, panss3, panss4):
    symptoms=df.loc[:, panss1:panss2].columns
    symptoms2=df.loc[:, panss3:panss4].columns
    div5=df.loc[:,'DMN':'AUD'].columns
    div3=df.loc[:,'wstr':'ast'].columns
    indices=symptoms.append((symptoms2, div5,div3))
    return indices

indices_bl=make_indices(dopa_data,'marder_dep','marder_pos', 'p1', 'g16')
bl_corr_data = (dopa_data.loc[dopa_data['patient'] == 1].loc[:,indices_bl]).dropna(axis=1)
[bl_corr, bl_p] = cb.cor_matrix(bl_corr_data, 'pearson')
bl_corr.to_pickle('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/pickled_databases/baseline_raw_corr.pkl')
bl_p.to_pickle('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/pickled_databases/baseline_raw_p.pkl')

masked_bl = pd.DataFrame(np.asarray(bl_p<0.05)*np.asarray(bl_corr), index = bl_corr.index, columns = bl_corr.columns)

response_data['cg7']

# Follow Up correlation matrix (n=20)
indices_ch=make_indices(response_data,'cmarder_dep','cmarder_pos', 'cp1', 'cg16')
ch_corr_data = (response_data.loc[:, indices_ch]).dropna(axis=1)
[ch_corr, ch_p] = cb.cor_matrix(ch_corr_data, 'pearson')
ch_corr.to_pickle('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/pickled_databases/change_raw_corr.pkl')
ch_p.to_pickle('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/pickled_databases/change_raw_p.pkl')
masked_ch = pd.DataFrame(np.asarray(ch_p<0.05)*np.asarray(ch_corr), index = ch_corr.index, columns = ch_corr.columns)

# Symptom modularity
def reorder_corr(dataframe, numberPartitions, consensusMatrixThreshold,LouvainMethod, gamma, seed):
    dataframe_np = dataframe.as_matrix()
    module_dict= bct.modularity_louvain_und(dataframe_np, gamma=1, hierarchy=False, seed=0)[0]
    [ignore1, ignore2, module_dict] = cb.consensus_clustering_louvain(np.asarray(dataframe),numberPartitions, consensusMatrixThreshold,LouvainMethod, gamma, seed)
    a=list(module_dict.values())
    b=[]
    for i in range(len(a)): b.append(a[i][0][0])
    new_order = np.concatenate(b)
    ordered_matrix = (dataframe_np[new_order,:])[:,new_order]
    return (ordered_matrix, new_order, module_dict)


# numberPartitions = 500
# gamma = 1.0
# LouvainMethod = 'negative_sym'
# consensusMatrixThreshold = 0.5

# NB give very similar result if using more similar method to the brain networks :
numberPartitions = 500
gamma = 1
LouvainMethod = 'modularity'
consensusMatrixThreshold = 0.5
seed=3
inputmatrix = bl_corr.loc['p1':'g16','p1':'g16']
inputmatrix2 = bct.threshold_proportional(np.asarray(inputmatrix), 0.5)
inputmatrix3 = pd.DataFrame(inputmatrix2, index = inputmatrix.index, columns = inputmatrix.columns)

# ch_ordered_matrix, ch_new_order, ch_module_dict = reorder_corr(ch_corr.loc['cp1':'cg16','cp1':'cg16'], 
#                                                                 numberPartitions, consensusMatrixThreshold,LouvainMethod, gamma)
                                                                


bl_ordered_matrix, bl_new_order, bl_module_dict = reorder_corr(inputmatrix3, 
                                                                numberPartitions, consensusMatrixThreshold,LouvainMethod, gamma, seed)                                                        

# Baseline
plt.rcParams['figure.figsize'] = [7, 7]
bl_tick_labels = (bl_corr.loc[:,'p1':'g16'].columns)[bl_new_order]
bl_mod_change = np.where(bl_new_order[:-1] >= bl_new_order[1:])[0]


sns.heatmap(bl_corr.loc['p1':'g16','p1':'g16'].iloc[bl_new_order,bl_new_order],xticklabels=bl_tick_labels, yticklabels=bl_tick_labels, cmap="RdBu_r", center=0)
# If using original louvain: for c in (np.append(1,bl_mod_change+1)):
for c in bl_mod_change+1:
    plt.axvline(x=c)
    plt.axhline(y=c)
plt.savefig('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/figures/graphs/bl_symptoms_correl.png', dpi=400)

# Change
ch_tick_labels = (ch_corr.loc[:,'cp1':'cg16'].columns)[ch_new_order]
ch_mod_change = np.where(ch_new_order[:-1] >= ch_new_order[1:])[0]
sns.heatmap(ch_ordered_matrix, xticklabels=ch_tick_labels, yticklabels=ch_tick_labels, cmap="RdBu_r", center=0)
for c in (ch_mod_change+1):
    plt.axvline(x=c)
    plt.axhline(y=c)

# Correlation Heatmaps
plt.rcParams['figure.figsize'] = [5, 5]
sns.heatmap(bl_corr.loc['p1':'g16','DMN':'AUD'], cmap="RdBu_r", vmin = -1, vmax=1, center=0)
sns.heatmap(bl_corr.loc['marder_dep':'marder_pos','DMN':'AUD'], cmap="RdBu_r", vmin = -0.6, vmax=0.6, center=0)
sns.heatmap(bl_corr.loc['p1':'g16','wstr':'ast'], cmap="RdBu_r", vmin = -1, vmax=1, center=0)
sns.heatmap(masked_bl.loc['p1':'g16','DMN':'AUD'], cmap="RdBu_r", vmin = -1, vmax=1, center=0)
sns.heatmap(masked_bl.loc['p1':'g16','wstr':'ast'], cmap="RdBu_r", vmin = -1, vmax=1, center=0)


# Compare martinez and resitng state
#calculations done in R compare_coeffs.rmd
plt.rcParams['figure.figsize'] = [10, 8]
subdivs=['DMN','SMh', 'CON','DAT', 'AUD', 'lst', 'ast', 'smst']
sns.heatmap(bl_corr.loc[subdivs,subdivs[::-1]], cmap="Reds", vmin = 0, vmax=1, center=0.5, cbar=True)
plt.axhline(y=5)
plt.axvline(x=3)
plt.savefig('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/figures/graphs/ki_correl.png', dpi=400)

plt.rcParams['figure.figsize'] = [10, 7]
p=pd.read_csv('/Users/robmcc/Documents/academic/DOPA_symptoms/results/kis_correlation_signif.csv')
z_data=pd.read_csv('/Users/robmcc/Documents/academic/DOPA_symptoms/results/kis_correlation_z.csv')
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
plt.savefig('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/figures/graphs/correl_comparison.png', dpi=400)


# Baseline ki-symp reordered
plt.rcParams['figure.figsize'] = [10, 15]
bl_indices_reorder = bl_p.loc[:,'p1':'g16'].columns[bl_new_order]
# bl_indices_reorder = bl_p.loc[:,'p1':'g16'].columns
# bl_indices_reorder = ['p1','p3', 'p5','p6','n7', 'g1', 'g9', 'g12', 'n1','n2', 'n3', 'n4', 'n6', 'g7', 'g16', 'p2','n5', 'g5', 'g11', 'g13', 'g15', 'g10', 'p4','p7', 'g8', 'g14', 'g2','g3', 'g4', 'g6']
x=np.where(bl_p.loc[bl_indices_reorder,'DMN':'AUD']<0.05)[1]
y=np.where(bl_p.loc[bl_indices_reorder,'DMN':'AUD']<0.05)[0]
sns.heatmap(bl_corr.loc[bl_indices_reorder,'DMN':'AUD'], cmap="RdBu_r", vmin = -0.6, vmax=0.6, center=0)
plt.yticks(fontsize = 15,rotation=0)
plt.xticks(fontsize = 15,rotation=0)
# for i in range(len(x)):
#     plt.text(x[i]+0.45,y[i]+0.9,'*', fontsize=26, color = 'white')
for c in (bl_mod_change+1):
#for c in (7,14):
#for c in (8,15,22,26 ):
    plt.axhline(y=c)
plt.savefig('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/figures/graphs/bl_symptomki_correl50.png', dpi=400)


# Change ki-symnp reordered
ch_indices_reorder = ch_p.loc[:,'cp1':'cg16'].columns[bl_new_order]
#ch_indices_reorder = ch_p.loc[:,'cp1':'cg16'].columns
#ch_indices_reorder = ['cp1','cp3', 'cp5','cp6','cn7', 'cg1', 'cg9', 'cg12', 'cn1','cn2', 'cn3', 'cn4', 'cn6', 'cg7', 'cg16', 'cp2','cn5', 'cg5', 'cg11', 'cg13', 'cg15', 'cg10', 'cp4','cp7', 'cg8', 'cg14', 'cg2','cg3', 'cg4', 'cg6']
x=np.where(ch_p.loc[ch_indices_reorder,'DMN':'AUD']<0.05)[1]
y=np.where(ch_p.loc[ch_indices_reorder,'DMN':'AUD']<0.05)[0]
sns.heatmap(ch_corr.loc[ch_indices_reorder,'DMN':'AUD'],yticklabels=ch_indices_reorder, cmap="RdBu_r", vmin = -0.6, vmax=0.6, center=0)
plt.yticks(fontsize = 15,rotation=0)
plt.xticks(fontsize = 15,rotation=0)
# for i in range(len(x)):
#     plt.text(x[i]+0.45,y[i]+0.9,'*', fontsize=26, color = 'white')
for c in (bl_mod_change+1):
#for c in (7,14):
#for c in (8,16,23,27 ):
    plt.axhline(y=c)
plt.savefig('/Users/robmcc/Documents/academic/DOPA_Symptoms/results/figures/graphs/ch_symptomki_correl50.png', dpi=400)



sns.heatmap(ch_corr.loc['cp1':'cg16','DMN':'AUD'], cmap="RdBu_r", vmin = -1, vmax=1,center=0)
sns.heatmap(ch_corr.loc['cp1':'cg16','wstr':'ast'], cmap="RdBu_r", vmin = -1, vmax=1,center=0)
sns.heatmap(masked_ch.loc['cp1':'cg16','DMN':'AUD'], cmap="RdBu_r", vmin = -1, vmax=1,center=0)
sns.heatmap(masked_ch.loc['cp1':'cg16','wstr':'ast'], cmap="RdBu_r", vmin = -1, vmax=1,center=0)

ch_corr['cg7']
ch_corr['cg7']

dopa_data['cg7']


# Regressing out whole striatum & PANSS total
symptoms = dopa_data.loc[:, 'p1':'g16'].columns
symptoms_change = response_data.loc[:, 'cp1':'cg16'].columns
subdivs = ['DMN', 'SMh', 'CON', 'DAT', 'AUD', 'lst', 'ast', 'smst']
par_corr_wstr_t = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
par_corr_wstr_p = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
par_corr_panss_t = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
par_corr_panss_p = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
par_corr_both_t = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
par_corr_both_p = pd.DataFrame(index=symptoms, columns=subdivs, dtype=float)
ch_par_corr_wstr_t = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)
ch_par_corr_wstr_p = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)
ch_par_corr_panss_t = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)
ch_par_corr_panss_p = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)
ch_par_corr_both_t = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)
ch_par_corr_both_p = pd.DataFrame(index=symptoms_change, columns=subdivs, dtype=float)

wstr_ki = patient_data.loc[:, "wstr"]
wstr_ki_ch = response_data.loc[:, "wstr"]
panss_total = patient_data.loc[:, "panss_total"]
panss_total_ch = response_data.loc[:, "panss_total_ch"]
for subdiv in subdivs:
    for symptom, symptom_change in zip(symptoms, symptoms_change):
        subdiv_kis = patient_data.loc[:, subdiv]
        ch_subdiv_kis = response_data.loc[:, subdiv]
        symptom_scores = patient_data.loc[:, symptom]
        ch_symptom_scores = response_data.loc[:, symptom_change]
        # Run bl regression                                         
        regression_data = pd.concat([symptom_scores, subdiv_kis, wstr_ki, panss_total], axis=1)
        regression_data.columns = ["symp", "subdiv", "wstr", "panss_total"]
        wstr_model = sm.ols(formula="symp~subdiv+wstr", data=regression_data).fit()
        panss_model = sm.ols(formula="symp~subdiv+panss_total", data=regression_data).fit()
        both_model = sm.ols(formula="symp~subdiv+wstr+panss_total", data=regression_data).fit()
        #run change regression
        regression_data_ch = pd.concat([ch_symptom_scores, ch_subdiv_kis, wstr_ki_ch, panss_total_ch], axis=1)
        regression_data_ch.columns = ["ch_symp", "ch_subdiv", "ch_wstr", "ch_panss_total"]
        ch_wstr_model = sm.ols(formula="ch_symp~ch_subdiv+ch_wstr", data=regression_data_ch).fit()
        ch_panss_model = sm.ols(formula="ch_symp~ch_subdiv+ch_panss_total", data=regression_data_ch).fit()
        ch_both_model = sm.ols(formula="ch_symp~ch_subdiv+ch_wstr+ch_panss_total", data=regression_data_ch).fit()
        #fill matrix
        par_corr_wstr_t.loc[symptom, subdiv] = wstr_model.tvalues[1]
        par_corr_wstr_p.loc[symptom, subdiv] = wstr_model.pvalues[1]
        par_corr_panss_t.loc[symptom, subdiv] = panss_model.tvalues[1]
        par_corr_panss_p.loc[symptom, subdiv] = panss_model.pvalues[1]
        par_corr_both_t.loc[symptom, subdiv] = both_model.tvalues[1]
        par_corr_both_p.loc[symptom, subdiv] = both_model.pvalues[1]
        ch_par_corr_wstr_t.loc[symptom_change, subdiv] = ch_wstr_model.tvalues[1]
        ch_par_corr_wstr_p.loc[symptom_change, subdiv] = ch_wstr_model.pvalues[1]
        ch_par_corr_panss_t.loc[symptom_change, subdiv] = ch_panss_model.tvalues[1]
        ch_par_corr_panss_p.loc[symptom_change, subdiv] = ch_panss_model.pvalues[1]
        ch_par_corr_both_t.loc[symptom_change, subdiv] = ch_both_model.tvalues[1]
        ch_par_corr_both_p.loc[symptom_change, subdiv] = ch_both_model.pvalues[1]

masked_wstr = pd.DataFrame(np.asarray(par_corr_wstr_p<0.05)*np.asarray(par_corr_wstr_t), index = par_corr_wstr_t.index, columns = par_corr_wstr_t.columns)
masked_panss = pd.DataFrame(np.asarray(par_corr_panss_p<0.05)*np.asarray(par_corr_panss_t), index = par_corr_panss_t.index, columns = par_corr_panss_t.columns)
masked_both = pd.DataFrame(np.asarray(par_corr_both_p<0.05)*np.asarray(par_corr_both_t), index = par_corr_both_t.index, columns = par_corr_both_t.columns)
ch_masked_wstr = pd.DataFrame(np.asarray(ch_par_corr_wstr_p<0.05)*np.asarray(ch_par_corr_wstr_t), index = ch_par_corr_wstr_t.index, columns = ch_par_corr_wstr_t.columns)
ch_masked_panss = pd.DataFrame(np.asarray(ch_par_corr_panss_p<0.05)*np.asarray(ch_par_corr_panss_t), index = ch_par_corr_panss_t.index, columns = ch_par_corr_panss_t.columns)
ch_masked_both = pd.DataFrame(np.asarray(ch_par_corr_both_p<0.05)*np.asarray(ch_par_corr_both_t), index = ch_par_corr_both_t.index, columns = ch_par_corr_both_t.columns)

# Regressed out Heatmaps
sns.heatmap(masked_wstr.loc['p1':'g16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(masked_wstr.loc['p1':'g16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(masked_panss.loc['p1':'g16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(masked_panss.loc['p1':'g16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(masked_both.loc['p1':'g16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(masked_both.loc['p1':'g16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)

sns.heatmap(ch_masked_wstr.loc['cp1':'cg16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(ch_masked_wstr.loc['cp1':'cg16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(ch_masked_panss.loc['cp1':'cg16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(ch_masked_panss.loc['cp1':'cg16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(ch_masked_both.loc['cp1':'cg16', 'DMN':'AUD'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)
sns.heatmap(ch_masked_both.loc['cp1':'cg16', 'lst':'smst'], cmap="RdBu_r", vmin = -5, vmax=5, center=0)



negative_bl = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7']
affect_bl = set(['p4', 'p5', 'p7', 'g2', 'g3', 'g4', 'g14'])

sns.regplot(x=affect_cluster, y=response_data.loc[:,'CON'])
sns.regplot(x=negative_cluster, y=response_data.loc[:,'CON'])
sns.regplot(x=response_data.loc[:,'cn4'], y=response_data.loc[:,'DAT'])
cluster1 = pd.DataFrame(np.asarray(response_data.loc[:, 'p1':'g16'])-np.asarray(response_data.loc[:, 'fp1':'fg16']))

col_list = ['panss_total_ch', 'insight_cluster_ch', 'positive_cluster_ch', 'negative_cluster_ch', 'affect_cluster_ch', 'DMN', 'SMh', 'CON', 'DAT', 'AUD']
r,p=cb.cor_matrix(response_data.loc[:,col_list ], 'pearson')
sns.heatmap(r, cmap="RdBu_r", vmin = -1, vmax=1, center=0)

sns.scatterplot(x=response_data['SMh'], y=response_data['cg7'])
sns.scatterplot(x=patient_data['g7'],y=patient_data['SMh'])
sns.scatterplot(x=patient_data['p3'],y=patient_data['AUD'])

sns.regplot(x=patient_data[negative_bl].sum(axis=1),y=patient_data['DMN'])

patient_data.to_csv('../../results/data_for_r/baseline.csv')
response_data.to_csv('../../results/data_for_r/change.csv')

from scipy.stats import pearsonr
pearsonr(response_data['AUD'], response_data['p3'])