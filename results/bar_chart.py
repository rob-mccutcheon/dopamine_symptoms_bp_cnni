# Load required modules
import pandas as pd
import numpy as np
from scipy import stats, linalg
import statsmodels.formula.api as sm
from matplotlib import pyplot as plt

pet_csv_path = ('/pet_network_kis/scripted.csv')

clinical_data = pd.read_csv('/results/copies_of_local/clinical.csv')

pet_data = pd.read_csv(pet_csv_path, names=['subject_id', 'DMN', 'SMh', 'SMm', 'VIS', 'FPN', 'CPN', 'RST', 'CON', 'DAT', 'AUD', 'VAT','SAL',' None'], index_col=None)
idx = clinical_data.columns.get_loc('g16')
clinical_data.insert(idx+1, 'panss_total', clinical_data.loc[:, 'p1':'g16'].sum(axis=1, skipna=False))
clinical_data.insert(idx+1, 'pos_total', clinical_data.loc[:, 'p1':'p7'].sum(axis=1, skipna=False))
dopa_data = clinical_data.join(pet_data.set_index('subject_id'), on = 'subject_id')
dopa_data=dopa_data.loc[dopa_data['include']==1]

patient_data=dopa_data[dopa_data['patient']==1]

xlabels = list(patient_data.columns[patient_data.columns.get_loc('DMN'):])+['ast', 'lst', 'smst','wstr']
dopa_data.columns[dopa_data.columns.get_loc('DMN'):]


graph_data= dopa_data.loc[(dopa_data['response'] < 3)| (dopa_data['patient'] == 0)]
dopa_melted = pd.melt(graph_data, id_vars = ['subject_id', 'response'], value_vars = xlabels)

# Dataframe To be filled
subdivisions = ['DMN', 'SMh', 'CON', 'DAT', 'AUD']
groups = ['Control', 'Responder', 'Non-Responder',]
df = pd.DataFrame({'Subdivision' : 3*subdivisions,
                  'Group' : 5*['Control']+5*['Responder']+ 5*['Non-Responder'],
                  'Value' : 'NA',
                  'Error' : 'NA'})

# Change melted response labels
melted_replace = [99,1,2]
i=0
for number in melted_replace:
    dopa_melted.loc[dopa_melted['response']==number, 'response'] = groups[i]
    i=i+1


for subdivision in subdivisions:
    for group in groups:
        #calcualte value
        value_result = np.mean(dopa_melted.loc[(dopa_melted['variable']==subdivision) & (dopa_melted['response']==group), 'value'])
        error_result = stats.sem(dopa_melted.loc[(dopa_melted['variable']==subdivision) & (dopa_melted['response']==group), 'value'])
        #Fill dataframe
        df.loc[(df['Subdivision']==subdivision) & (df['Group']==group), 'Value'] = value_result
        df.loc[(df['Subdivision']==subdivision) & (df['Group']==group), 'Error'] = error_result



#### PATIENT CONTROL GRAPH

dopa_melted = pd.melt(graph_data, id_vars = ['subject_id', 'patient'], value_vars = xlabels)

# Dataframe To be filled
subdivisions = ['DMN', 'SMh', 'CON', 'DAT', 'AUD']
groups = ['Control', 'Patient']
df = pd.DataFrame({'Subdivision' : 2*subdivisions,
                  'Group' : 5*['Control']+5*['Patient'],
                  'Value' : 'NA',
                  'Error' : 'NA'})

# Change melted response labels
melted_replace = [0,1]
i=0
for number in melted_replace:
    dopa_melted.loc[dopa_melted['patient']==number, 'patient'] = groups[i]
    i=i+1


for subdivision in subdivisions:
    for group in groups:
        #calcualte value
        value_result = np.mean(dopa_melted.loc[(dopa_melted['variable']==subdivision) & (dopa_melted['patient']==group), 'value'])
        error_result = stats.sem(dopa_melted.loc[(dopa_melted['variable']==subdivision) & (dopa_melted['patient']==group), 'value'])
        #Fill dataframe
        df.loc[(df['Subdivision']==subdivision) & (df['Group']==group), 'Value'] = value_result
        df.loc[(df['Subdivision']==subdivision) & (df['Group']==group), 'Error'] = error_result

###########


def grouped_barplot(df, cat,subcat, val , err):
    u = df[cat].unique()
    x = np.arange(len(u))
    subx = df[subcat].unique()
    offsets = (np.arange(len(subx))-np.arange(len(subx)).mean())/(len(subx)+1.)
    width= np.diff(offsets).mean()
    for i,gr in enumerate(subx):
        dfg = df[df[subcat] == gr]
        plt.bar(x+offsets[i], dfg[val].values, width=width, 
                label="{}".format(gr), yerr=dfg[err].values,color = colors[i])
    plt.xlabel(cat)
    plt.ylabel('Ki')
    plt.xticks(x, u)
    plt.legend()

#T-Test
stats.ttest_ind(dopa_data.loc[dopa_data['response']==1, 'DMN'], dopa_data.loc[dopa_data['patient']==0, 'DMN'], equal_var=False)

np.std(dopa_data.loc[dopa_data['response']==1, 'DMN'])
np.mean(dopa_data.loc[dopa_data['patient']==0, 'DMN'])

dopa_data[dopa_data['patient']==1]
dopa_data['patient']
subdivisions = ['DMN', 'SMh', 'CON', 'DAT', 'AUD', 'wstr', 'lst', 'ast', 'smst']
for network in subdivisions:
        print(stats.ttest_ind(dopa_data.loc[dopa_data['patient']==0, network], dopa_data.loc[dopa_data['response']==1, network], equal_var=False))
       #print(stats.ttest_ind(dopa_data.loc[dopa_data['patient']==1, network], dopa_data.loc[dopa_data['patient']==0, network]))

network='scan_gap'
print(stats.ttest_ind(dopa_data.loc[dopa_data['patient']==0, network], dopa_data.loc[dopa_data['patient']==1, network], equal_var=False))