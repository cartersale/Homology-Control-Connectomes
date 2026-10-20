import pandas as pd, numpy as np
d=pd.read_csv('q1/set_q1.csv.gz')
fams=['degree_strength','eigenvector_centrality','cycle_participation','random','top_avg_ctrl','betweenness']
c1=d[d.c==0.1]
# per subject per k mean, then k average, then subject mean
def kavg(x,ks,col):
    x=x[x.k.isin(ks)]
    g=x.groupby(['family','scale','subject_id','k'])[col].mean().groupby(['family','scale','subject_id']).mean().groupby(['family','scale']).mean().unstack()
    return g
print('loading all k'); print(kavg(c1[c1.family.isin(fams)],[1,2,3,5],'loading').round(3))
print('perron_share k2-5'); print(kavg(c1[c1.family.isin(fams)],[2,3,5],'perron_share').round(3))
print('q1 k2-5'); print(kavg(c1[c1.family.isin(fams)],[2,3,5],'q1').round(3))
for c in [0.01,0.1,1.0]:
  x=d[(d.c==c)]
  s=x.groupby(['family','scale','subject_id','k'])['q1'].mean()
  r=s.xs('random',level='family')
  for fam in ['degree_strength','cycle_participation']:
    f=s.xs(fam,level='family')
    diff=(f-r)
    for ks in [[1,2,3,5],[2,3,5]]:
      dd=diff[diff.index.get_level_values('k').isin(ks)].groupby(['scale','subject_id']).mean()
      print(c,fam,ks,'q1-rand',dd.groupby('scale').mean().round(3).tolist(),(dd>0).groupby('scale').sum().tolist())
# q1_perp degree-random k2-5
x=c1; s=x.groupby(['family','scale','subject_id','k'])['q1_perp'].mean()
diff=s.xs('degree_strength',level='family')-s.xs('random',level='family')
for ks in [[1,2,3,5],[2,3,5]]:
  dd=diff[diff.index.get_level_values('k').isin(ks)].groupby(['scale','subject_id']).mean()
  print('perp deg-rand',ks,dd.groupby('scale').mean().round(3).tolist(),(dd>0).groupby('scale').sum().tolist())
  # per k counts
for k in [2,3,5]:
  dd=diff.xs(k,level='k'); print(k,(dd>0).groupby('scale').sum().tolist())
