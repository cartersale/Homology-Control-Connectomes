import pandas as pd, numpy as np
from scipy.stats import spearmanr
d=pd.read_csv('/Users/cartersale/Research/projects/higher_order_connectomes/results/revision/q1/set_q1.csv.gz')
base=['betweenness','cycle_participation','degree_strength','eigenvector_centrality','participation_coefficient','triangle_participation','cycle_criticality','degree_normalized_cycle_participation','tetrahedron_participation','random']
out=[]
for c in [0.01,0.1,1.0]:
    for sc in [68,114,219]:
        g=d[(d.c==c)&(d.scale==sc)]
        for name,sub in [('all families, all k',g),('all families, k>=2',g[g.k>=2]),('rankings+random (no cf/matched), all k',g[g.family.isin(base)]),('rankings+random, k>=2',g[g.family.isin(base)&(g.k>=2)])]:
            out.append((c,sc,name,'pooled',spearmanr(sub.loading,sub.q1)[0]))
        # within subject, all k pooled, median
        for name,sub in [('all families, all k',g),('all families, k>=2',g[g.k>=2])]:
            r=sub.groupby('subject_id').apply(lambda x: spearmanr(x.loading,x.q1)[0])
            out.append((c,sc,name,'within-subj median',r.median()))
        r=g.groupby(['subject_id','k']).apply(lambda x: spearmanr(x.loading,x.q1)[0]).groupby('k').median()
        for k,v in r.items(): out.append((c,sc,f'all families, k={k}','within-subj(k) median',v))
    g=d[d.c==c]
    out.append((c,'all','all families, all k','pooled',spearmanr(g.loading,g.q1)[0]))
    out.append((c,'all','all families, k>=2','pooled',spearmanr(g[g.k>=2].loading,g[g.k>=2].q1)[0]))
o=pd.DataFrame(out,columns=['c','scale','sets','pooling','rho'])
pd.set_option('display.width',200); pd.set_option('display.max_rows',500)
print(o.round(3).to_string())
o.to_csv('/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables/ls_q1_candidates.csv',index=False)
