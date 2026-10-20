import pandas as pd, numpy as np
d=pd.read_csv('q1/set_q1.csv.gz')
d=d[d.k.isin([2,3,5])]
crit=['betweenness','cycle_criticality','cycle_participation','degree_normalized_cycle_participation','degree_strength','eigenvector_centrality','participation_coefficient','tetrahedron_participation','triangle_participation','cycle_resid','top_avg_ctrl','top_modal_gu','top_modal_ct','low_perron','random']
for name,fs in [('crit+rand',crit),('all',None)]:
  x=d if fs is None else d[d.family.isin(fs)]
  for c in [0.01,0.1,1.0]:
    y=x[x.c==c]
    pooled=y.groupby(['scale','subject_id']).apply(lambda g: g.loading.corr(g.q1,method='spearman')).groupby('scale').median()
    perk=y.groupby(['scale','subject_id','k']).apply(lambda g: g.loading.corr(g.q1,method='spearman')).groupby(['scale','subject_id']).mean().groupby('scale').median()
    print(name,c,'pooled',pooled.round(2).tolist(),'perk-avg',perk.round(2).tolist())
