# plot correlation 
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import kendalltau
import numpy as np 

TASK='valacc'

path = "./results_valacc_10p.json"
with open(path) as f:
    data = json.load(f)
gt_list = data['gt']
pd_list = data['pd']

sns.set_theme(style="whitegrid")
ax = sns.scatterplot(x=gt_list, y=pd_list)
ax.set_title(f'Correlation Plot of {TASK}')
ax.set_xlabel('Ground Truth')
ax.set_ylabel('Predicted')

# calculate kendall tau and mse 
tau, p_value = kendalltau(gt_list, pd_list)
mse = ((np.array(gt_list) - np.array(pd_list)) ** 2).mean()

# add text to the plot
textstr = '\n'.join((
    r'$\tau=%.2f$' % (tau, ),
    r'$mse=%.2f$' % (mse, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(f'violinplot_10p_{TASK}.png')
plt.clf()

# draw another figure that filterout the gt under 20 
gt_list_filtered, pd_list_filtered = [], []
for gt, pd in zip(gt_list, pd_list):
    if gt > 20:
        gt_list_filtered.append(gt)
        pd_list_filtered.append(pd)

sns.set_theme(style="whitegrid")
ax = sns.scatterplot(x=gt_list_filtered, y=pd_list_filtered)
ax.set_title(f'Correlation Plot of {TASK} > 20')
ax.set_xlabel('Ground Truth')
ax.set_ylabel('Predicted')

# calculate kendall tau
tau, p_value = kendalltau(gt_list_filtered, pd_list_filtered)
mse = ((np.array(gt_list) - np.array(pd_list)) ** 2).mean()

# add text to the plot
textstr = '\n'.join((
    r'$\tau=%.2f$' % (tau, ),
    r'$mse=%.2f$' % (mse, )))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.1, 0.9, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig(f'violinplot_10p_filtered_{TASK}.png')
plt.clf()
