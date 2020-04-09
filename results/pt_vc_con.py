import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.stats import sem

#  Amend parser.read statement and amend project_details.txt file depending on where you are running
parser.read('/project_details.txt')
main_directory = parser.get('project_details', 'main_directory')
striatal_map_dir = ('%sresults/indiv_striatum_probability_maps/scripted/numpy' % main_directory)
num_networks = int(parser.get('project_details', 'num_networks'))
img_dim = (loads(parser.get("project_details", "image_dimension")))
img_dim = [int(i)for i in img_dim]
clinical_data = pd.read_csv('%sresults/copies_of_local/clinical.csv' % main_directory)


clinical_data = pd.read_csv('results/clinical.csv')
striatal_map_dir = 'DOPA_symptoms/results/numpy'

# add new columns
network_ids = [0, 1, 4, 7, 8, 9, 12]
network_ids = [0, 1,2,3, 4,5,6, 7, 8, 9,10,11, 12]
for network_id in network_ids:
    clinical_data[f'network_{network_id}'] = np.nan

# Calculate striatum subdiv strength
for subject in range(1, 53):
    str_map = np.load(f'{striatal_map_dir}/striatum_only_{subject:03}.npy')
    for network in network_ids:
        network_strength = np.nansum(str_map[:, :, :, network])
        clinical_data.loc[clinical_data['subject_id']==subject, f'network_{network}'] = network_strength

# Compare patients and controls
patient_df = clinical_data[clinical_data['patient']==1]
patient_df = patient_df[patient_df['include']==1]
control_df = clinical_data[clinical_data['patient']==0]
control_df = control_df[control_df['include']==1]

pvalues = []
tvalues = []
mean_pt = []
mean_con = []
se_pt = []
se_con=[]
for network in network_ids:
    pt_strengths = patient_df[f'network_{network}']
    con_strengths = control_df[f'network_{network}']
    mean_pt.append(np.mean(pt_strengths))
    mean_con.append(np.mean(con_strengths))
    se_pt.append(sem(pt_strengths))
    se_con.append(sem(con_strengths))
    t,p = ttest_ind(pt_strengths, con_strengths)
    pvalues.append(p)
    tvalues.append(t)

Value = np.concatenate([np.array(mean_con)[[0,1,7,8,9]], np.array(mean_pt)[[0,1,7,8,9]]])
Error = np.concatenate([np.array(se_con)[[0,1,7,8,9]], np.array(se_pt)[[0,1,7,8,9]]])


columns = ['Subdivision', 'Group', 'Value', 'Error']
Subdivision = 2*['DMN', 'SMN', 'CON', 'DAT', 'AUD']
Group = 5*['Control']+ 5*['Patient']
Data = {'Subdivision':Subdivision,
        'Group':Group,
        'Value':Value,
        'Error':Error}
df =  pd.DataFrame(Data, columns = columns)

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

cat = "Subdivision"
subcat = "Group"
val = "Value"
err = "Error"
plt.rcParams['figure.figsize'] = [14, 10]
colors = ['#A9B5FF', '#FF807F', '#FF4644']
grouped_barplot(df, cat, subcat, val, err )
plt.savefig('/results/figures/graphs/connectivity_bar_chart.png', dpi=400)



#  determine spatial spread of maps for pts and controls

# make gradient maps
x1 = np.arange(1,92)
x2 = np.vstack([x1]*109).T
x_grad = np.dstack([x2]*91)

y1 = np.arange(1,110)
y2 = np.vstack([y1]*91)
y_grad = np.dstack([y2]*91)

z_grad = np.tile(np.arange(1,92), (91,109,1))


# calculate overlap
con_idx = (control_df['subject_id'].values-1).astype(int)
pt_idx = (patient_df['subject_id'].values-1).astype(int)

for network in [0,1,7,8,9]:
    grad_dic = {'x':[], 'y':[], 'z':[], 'sub':[]}
    for subject in range(1, 53):
        str_map = np.load(f'{striatal_map_dir}/striatum_only_{subject:03}.npy')
        grad_dic['sub'].append(subject)
        for dim in ['x','y','z']:
            score = np.nansum(eval(f'{dim}_grad')*str_map[:,:,:,network])
            grad_dic[dim].append(score)
    print(network)
    for dim in ['x','y','z']: 
        pt_values=np.array(grad_dic[dim])[pt_idx]
        con_values=np.array(grad_dic[dim])[con_idx]
        t,p = ttest_ind(pt_values, con_values)
        print(t,p)
