import pandas as pd, numpy as np
R='/Users/cartersale/Research/projects/higher_order_connectomes/results/revision/'
OUT='/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables/S7_setlevel.tex'
S=[68,114,219]; C=[0.01,0.1,1.0]; K=[2,3,5]
f2=lambda x: f"{x:+.2f}".replace('+','$+$').replace('-','$-$') if np.isfinite(x) else '--'
def f2(x):
    if not np.isfinite(x): return '--'
    s=f"{x:.2f}"
    if s in('-0.00',): s='0.00'
    return s.replace('-','$-$')
cs=lambda c: {0.01:'0.01',0.1:'0.1',1.0:'1'}[c]
L=[]
def table(caption,label,colspec,header,rows,note=None,size='\\scriptsize'):
    L.append('% ---- '+label)
    L.append('% Caption: '+caption)
    if note: L.append('% Note: '+note)
    L.append(f'% Suggested font size: {size}')
    L.append(f'\\begin{{tabular}}{{{colspec}}}\\toprule')
    L.extend(header); L.append('\\midrule')
    L.extend(rows); L.append('\\bottomrule')
    L.append('\\end{tabular}\n')

# --- S7.1 node-level correspondence
n=pd.read_csv(R+'modal/node_level_correlations.csv'); n=n[n.c==0.1]
nm={'modal_gu':'Modal ctrl.\\ (Gu)','modal_ct':'Modal ctrl.\\ (cont.-time)','avg_ctrl':'Average ctrl.','perron2':'$v_1(i)^2$'}
rk={'degree_strength':'Strength','cycle_participation':'Cycle part.','eigenvector_centrality':'Eigenvector','betweenness':'Betweenness'}
rows=[]
for a in nm:
    for b in rk:
        cells=[]
        for s in S:
            r=n[(n.node_metric==a)&(n.ranking==b)&(n.scale==s)].iloc[0]
            cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
        rows.append(f"{nm[a]} & {rk[b]} & "+' & '.join(cells)+' \\\\')
    rows.append('\\addlinespace')
rows=rows[:-1]
table('Node-level Spearman correlation between single-node controllability scores and the ranking criteria (within subject, across regions; $c=0.1$). Median across subjects (number of subjects with $\\rho>0$).','tab:S7_nodelevel','llccc',
      ['Node score & Ranking & 68 & 114 & 219 \\\\'],rows,
      'Modal controllability and $v_1(i)^2$ do not depend on $c$; average controllability does (at $c=0.01$ its $\\rho$ with strength is 0.90 / 0.89 / 0.89).')

# --- S7.2 within-subject over random sets
w=pd.read_csv(R+'modal/within_subject_summary.csv')
preds=[('rho_q1_perron_loading','$\\rho(L_S,q_1)$'),('rho_q1_avg_ctrl','$\\rho(\\overline{\\mathrm{AC}}_S,q_1)$'),('rho_q1_modal_gu','$\\rho(\\overline{\\mathrm{MC}}_S,q_1)$'),
       ('rho_q1_q1_G','$\\rho(q_1(G_S),q_1)$'),('rho_q1_red_G','$\\rho(\\bar r_G,q_1)$'),
       ('prho_q1_redGperp|node','partial $\\rho(\\bar r_{G\\perp},q_1\\mid\\text{node})$'),('prho_q1_redF|node','partial $\\rho(\\bar r_F,q_1\\mid\\text{node})$')]
rows=[]
for c in C:
    rows.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{$c={cs(c)}$}}}} \\\\")
    for v,lab in preds:
        cells=[]
        for s in S:
            for k in K:
                r=w[(w.variable==v)&(w.scale==s)&(w.k==k)&(w.c==c)].iloc[0]
                cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
        rows.append(lab+' & '+' & '.join(cells)+' \\\\')
table('Within-subject Spearman correlations with $q_1$ across each subject\'s 100 random driver sets. Median across 70 subjects (subjects with $\\rho>0$, of 70). Node = $\\{L_S,\\ \\overline{\\mathrm{MC}}_S^{\\mathrm{Gu}},\\ \\overline{\\mathrm{AC}}_S\\}$.','tab:S7_withinsubj','l'+'c'*9,
 ['& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\','\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
  'Predictor '+''.join(f'& $k={k}$ ' for _ in S for k in K)+'\\\\'],rows,
 '$q_1(G_S)=\\lambda_1(G_S)/\\operatorname{tr}G_S$ for the $k\\times k$ driver Gram matrix of slow-mode profiles $x_i=(u_a(i)/\\sqrt{-2\\mu_a})_a$; $\\bar r_G$ = mean pairwise cosine of these profiles; $\\bar r_{G\\perp}$ = same with the Perron mode removed; $\\bar r_F$ = mean pairwise Frobenius cosine of single-node Gramians. $k=1$ not computed (no pairs).',size='\\tiny')

# --- S7.3 driver-Gram proxy (pooled)
g=pd.read_csv(R+'modal/driver_gram_proxy.csv')
rows=[]
for c in C:
    for k in K:
        cells=[]
        for s in S:
            r=g[(g.scale==s)&(g.k==k)&(g.c==c)].iloc[0]
            cells+= [f2(r.rho_q1G_q1raw), f2(r.median_q1G_minus_q1raw), f2(r.median_q1raw_minus_pshare)]
        rows.append(f"{cs(c)} & {k} & "+' & '.join(cells)+' \\\\')
    rows.append('\\addlinespace')
rows=rows[:-1]
table('Driver-Gram bounds on $q_1$, pooled over all driver sets (random, ranked and counterfactual) and subjects. $\\rho$: Spearman $\\rho(q_1(G_S),q_1)$; upper gap: median $q_1(G_S)-q_1$; lower gap: median $q_1-\\pi_S$, where $\\pi_S=v_1^\\top W v_1/\\operatorname{tr}W=(L_S/2c)/\\operatorname{tr}W$.','tab:S7_drivergram','cc'+'ccc'*3,
 ['& & \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\','\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}',
  '$c$ & $k$ '+'& $\\rho$ & upper & lower '*3+'\\\\'],rows,
 'Both gaps are $\\ge 0$ in every cell, i.e.\\ $\\pi_S\\le q_1\\le q_1(G_S)$. Here $q_1$ is computed on the raw $W$ (no $\\varepsilon$ floor; \\texttt{q1\\_raw}).')

# --- S7.4 communicability vs geodesic
rd=pd.read_csv(R+'q1/redundancy_summary.csv')
rows=[]
for c in C:
    for s in S:
        r=rd[(rd.scale==s)&(rd.c==c)].iloc[0]
        rows.append(f"{cs(c)} & {s} & {f2(r.rho_L)} & {f2(r.rho_comm)} & {f2(r.rho_geo_inv)} & {f2(r['prho_comm|L'])} & {f2(r['prho_geo_inv|L'])} & {f2(r['prho_geo_log|L'])} & {int(r.n_comm_better_geo_inv_given_L)}/70 & {r.p_comm_vs_geo_inv_given_L:.1e} \\\\")
    rows.append('\\addlinespace')
rows=rows[:-1]
table('Redundancy as shared slow-mode access vs adjacency. Within-subject Spearman $\\rho$ with $q_1$ across the 114 driver sets per subject (14 named criteria + 100 random), averaged over $k\\in\\{2,3,5\\}$, median across subjects. comm = mean pairwise driver-Gram cosine ($=\\bar r_G$); geo = mean pairwise shortest-path length with edge length $1/w$ or $-\\log w$.','tab:S7_comm_geo','cc'+'c'*8,
 ['& & \\multicolumn{3}{c}{raw $\\rho$} & \\multicolumn{3}{c}{partial $\\rho\\mid L_S$} & \\multicolumn{2}{c}{comm vs geo$_{1/w}$ $\\mid L_S$} \\\\','\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}',
  '$c$ & $N$ & $L_S$ & comm & geo$_{1/w}$ & comm & geo$_{1/w}$ & geo$_{-\\log w}$ & $n$ better & $p$ \\\\'],rows,
 '$n$ better: subjects with $|\\rho_{\\text{comm}\\mid L}|>|\\rho_{\\text{geo}\\mid L}|$; $p$: paired Wilcoxon on $|\\rho_{\\text{comm}\\mid L}|-|\\rho_{\\text{geo}\\mid L}|$ across subjects (uncorrected).')

# --- S7.5 modal sets vs cycle sets
o=pd.read_csv(R+'modal/set_overlap.csv'); o=o[o.c==0.1]
cf=pd.read_csv(R+'modal/cycle_vs_counterfactual.csv'); cf=cf[cf.c==0.1]
rows=[]
for a,al in [('top_modal_gu','Top-$k$ modal (Gu)'),('top_avg_ctrl','Top-$k$ average ctrl.'),('low_perron','Lowest $v_1(i)^2$')]:
    for b,bl in [('cycle_participation','cycle'),('degree_strength','degree'),('eigenvector_centrality','eigenvector')]:
        cells=[f2(o[(o.a==a)&(o.b==b)&(o.scale==s)&(o.k==k)].jaccard.iloc[0]) for s in S for k in K]
        rows.append(f"Jaccard: {al} vs {bl} & "+' & '.join(cells)+' \\\\')
rows.append('\\midrule')
for con,cl in [('cycle - top_modal_gu','cycle $-$ modal (Gu)'),('cycle - top_avg_ctrl','cycle $-$ avg.\\ ctrl.'),('cycle - low_perron','cycle $-$ lowest $v_1^2$')]:
    for t,tl in [('q1','$\\Delta q_1$'),('effective_rank','$\\Delta$ER'),('er_raw','$\\Delta$ER (no floor)')]:
        cells=[]
        for s in S:
            for k in K:
                r=cf[(cf.contrast==con)&(cf.target==t)&(cf.scale==s)&(cf.k==k)].iloc[0]
                cells.append(f"{f2(r['median'])} ({int(r.n_pos)}/70)")
        rows.append(f"{tl}: {cl} & "+' & '.join(cells)+' \\\\')
table('Counterfactual node-score sets vs cycle-participation sets ($c=0.1$). Top: mean Jaccard overlap across subjects. Bottom: paired difference cycle $-$ counterfactual set, median across subjects (subjects with difference $>0$, of 70).','tab:S7_modal_sets','l'+'c'*9,
 ['& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\','\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
  ''+''.join(f'& $k={k}$ ' for _ in S for k in K)+'\\\\'],rows,
 'ER = exp-entropy effective rank of the $\\varepsilon$-floored spectrum ($\\varepsilon=10^{-5}$, pipeline definition); ER (no floor) uses the raw spectrum. Overlap of modal/avg.-ctrl.\\ sets with ranked sets depends weakly on $c$ through average controllability; modal-set overlaps are 0.00 with cycle, degree and eigenvector sets at every $c$.',size='\\tiny')

# --- S7.6 ranked-set prediction (q1)
h=pd.read_csv(R+'modal/ranked_set_prediction_summary.csv'); h=h[(h.c==0.1)&(h.target=='q1')]
rows=[]
cells=[f2(h[(h.scale==s)&(h.k==k)&(h.model=='node3')].obs_median.iloc[0]) for s in S for k in K]
rows.append('Observed cycle $-$ degree $\\Delta q_1$ & '+' & '.join(cells)+' \\\\')
for m,ml in [('node3','node scores'),('node3+redGperp','node scores + $\\bar r_{G\\perp}$'),('pshare','Perron share $\\pi_S$'),('q1G','$q_1(G_S)$')]:
    cells=[]
    for s in S:
        for k in K:
            r=h[(h.scale==s)&(h.k==k)&(h.model==m)].iloc[0]
            cells.append(f"{f2(r.pred_median)} [{f2(r.rho_obs_pred_across_subjects)}]")
    rows.append(f"Predicted: {ml} & "+' & '.join(cells)+' \\\\')
table('Out-of-sample prediction of the cycle $-$ degree $q_1$ difference from linear models fit on each subject\'s 100 random sets ($c=0.1$). Median predicted difference across subjects [Spearman $\\rho$ between observed and predicted differences across subjects].','tab:S7_rankedpred','l'+'c'*9,
 ['& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\','\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
  ''+''.join(f'& $k={k}$ ' for _ in S for k in K)+'\\\\'],rows,None,size='\\tiny')

# --- S7.7 coverage
cb=pd.read_csv(R+'mechanism/coverage_by_criterion.csv'); cb=cb[cb.kgrp=='2+']
fams=[('betweenness','Betweenness'),('degree_strength','Degree/strength'),('cycle_participation','Cycle participation'),('cycle_resid','Cycle (degree-resid.)'),('eigenvector_centrality','Eigenvector centrality'),('top_avg_ctrl','Top-$k$ average ctrl.'),('participation_coefficient','Participation coef.'),('top_modal_gu','Top-$k$ modal ctrl.\\ (Gu)'),('low_perron','Lowest $v_1(i)^2$')]
rows=[]
for f,fl in fams:
    cells=[]
    for s in S:
        r=cb[(cb.family==f)&(cb.scale==s)].iloc[0]
        cells.append(f"{f2(r.mean_d_inv)} & {f2(r.pct_inv)} & {int(r.n_closer_inv)}/70")
    rows.append(fl+' & '+' & '.join(cells)+' \\\\')
table('Coverage distance (mean over non-driver regions of the $1/w$ shortest-path distance to the nearest driver) of criterion sets vs each subject\'s random sets, $k\\ge2$. $\\Delta$: set $-$ random (subject mean over $k$, then mean across subjects); pct: percentile among random sets; $n$: subjects with set closer than random.','tab:S7_coverage','l'+'ccc'*3,
 ['& \\multicolumn{3}{c}{68} & \\multicolumn{3}{c}{114} & \\multicolumn{3}{c}{219} \\\\','\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
  'Criterion '+'& $\\Delta$ & pct & $n$ '*3+'\\\\'],rows,
 'Coverage is independent of $c$. Hop-count and $-\\log w$ versions are in \\texttt{coverage\\_by\\_criterion.csv}.')
cr=pd.read_csv(R+'mechanism/coverage_rho.csv')
rk_=cr[cr.k>1].groupby(['scale','subject_id','c']).mean(numeric_only=True).reset_index()
rows=[]
for c in C:
    for s in S:
        g_=rk_[(rk_.scale==s)&(rk_.c==c)]
        cells=[]
        for col in ['rho_inv','prho_inv|L','rho_log','prho_log|L','rho_hop','prho_hop|L']:
            v=g_[col]; cells.append(f"{f2(v.median())} ({int((v>0).sum())}/70)")
        rows.append(f"{cs(c)} & {s} & "+' & '.join(cells)+' \\\\')
    rows.append('\\addlinespace')
rows=rows[:-1]
table('Within-subject Spearman $\\rho$(coverage distance, $q_1$) across driver sets, raw and partial given $L_S$; per-subject mean over $k\\in\\{2,3,5\\}$, median across subjects (subjects with $\\rho>0$).','tab:S7_coverage_rho','cc'+'c'*6,
 ['& & \\multicolumn{2}{c}{$1/w$} & \\multicolumn{2}{c}{$-\\log w$} & \\multicolumn{2}{c}{hops} \\\\','\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}',
  '$c$ & $N$ & raw & $\\mid L_S$ & raw & $\\mid L_S$ & raw & $\\mid L_S$ \\\\'],rows,
 'Summary computed by us from per-subject rows in \\texttt{coverage\\_rho.csv} (read-only), reproducing the aggregation in \\texttt{03\\_coverage\\_distance.py}.')

# --- S7.8 hub loading
hl=pd.read_csv(R+'mechanism/hub_loading_summary.csv')
stats=[('cos_s_v1','$\\cos(s,v_1)$'),('cos_As_v1','$\\cos(As,v_1)$'),('rho_logs_logm','$\\rho(s_i,m_i)$'),('jac_top5','Top-5 Jaccard ($s$ vs $v_1$)'),('L_deg_5','$L_S$, top-5 strength'),('L_max_5','$L_{\\max}$, top-5 by $v_1^2$'),('frac','$L_S/L_{\\max}$ ($k=5$)'),('strength_assort','Strength assort.\\ (bct)'),('l1','$\\ell_1$'),('gap','Gap $\\ell_1-\\ell_2$')]
rows=[]
for st,sl in stats:
    for s in S:
        cells=[]
        if st=='frac':
            real=None
            for nl in ['strength','geometric','weights']:
                a=hl[(hl.null==nl)&(hl.scale==s)&(hl.stat=='L_deg_5')].iloc[0]; b=hl[(hl.null==nl)&(hl.scale==s)&(hl.stat=='L_max_5')].iloc[0]
                real=a.real_mean/b.real_mean; cells.append(f"{f2(a.null_mean/b.null_mean)} (--)")
        else:
            for nl in ['strength','geometric','weights']:
                r=hl[(hl.null==nl)&(hl.scale==s)&(hl.stat==st)].iloc[0]; real=r.real_mean
                cells.append(f"{f2(r.null_mean)} ({int(r.n_real_higher)}/70)")
        rows.append((sl if s==68 else '')+f" & {s} & {f2(real)} & "+' & '.join(cells)+' \\\\')
    rows.append('\\addlinespace')
rows=rows[:-1]
table('Hub-loading identity $v_1(i)=s_i m_i/\\ell_1$: real connectomes vs null networks. Means across subjects (null: mean of each subject\'s 50-null mean); in parentheses, subjects with real $>$ own null mean (of 70).','tab:S7_hubloading','llcccc',
 ['Statistic & $N$ & Real & Strength null & Geometric null & Weight null \\\\'],rows,
 '$m_i=\\sum_j A_{ij}v_1(j)/s_i$. $\\rho(s_i,m_i)$ is Spearman across regions (computed on $\\log s,\\log m$). $L_S/L_{\\max}$ is the ratio of the across-subject means (computed by us from the two rows above; no per-subject count). Wilcoxon $p$ for each row is in \\texttt{hub\\_loading\\_summary.csv}.')
ha=pd.read_csv(R+'mechanism/hub_loading_assort_corr.csv')
rows=[]
for y,yl in [('rho_logs_logm','$\\rho(s_i,m_i)$'),('cos_s_v1','$\\cos(s,v_1)$'),('L_deg_5','$L_S$ (top-5 strength)'),('l1','$\\ell_1$'),('gap','Gap')]:
    cells=[]
    for s in S:
        r=ha[(ha.scale==s)&(ha.y==y)].iloc[0]; cells.append(f"{f2(r.rho_with_assort)} ({r.p:.2g})")
    rows.append(yl+' & '+' & '.join(cells)+' \\\\')
table('Across real subjects ($n=70$): Spearman $\\rho$ of strength assortativity (bct) with hub-loading quantities ($p$).','tab:S7_assortcorr','lccc',['& 68 & 114 & 219 \\\\'],rows,None)

# --- S7.9 L_S vs q1 candidates
cand=pd.read_csv('/private/tmp/claude-501/-Users-cartersale-Research-projects-higher-order-connectomes/cea7c7b4-c39c-4a27-8881-388281d3a80d/scratchpad/tables/ls_q1_candidates.csv')
cand['scale']=cand.scale.astype(str)
pool=pd.read_csv(R+'modal/pooled_spearman.csv'); pool=pool[(pool.target=='q1')&(pool.predictor=='perron_loading')]
xs=pd.read_csv(R+'modal/cycle_minus_degree_node_vs_set.csv')
rows=[]
def crow(lab,getter):
    rows.append(lab+' & '+' & '.join(f2(getter(c,s)) for c in C for s in S)+' \\\\')
crow('(a) Pooled, all sets, all $k$',lambda c,s: cand[(cand.c==c)&(cand.scale==str(s))&(cand.sets=='all families, all k')&(cand.pooling=='pooled')].rho.iloc[0])
crow('(b) Pooled, all sets, $k\\ge2$',lambda c,s: cand[(cand.c==c)&(cand.scale==str(s))&(cand.sets=='all families, k>=2')&(cand.pooling=='pooled')].rho.iloc[0])
crow('(c) Within-subj., all sets, $k\\ge2$',lambda c,s: cand[(cand.c==c)&(cand.scale==str(s))&(cand.sets=='all families, k>=2')&(cand.pooling=='within-subj median')].rho.iloc[0])
for k in [1,2,3,5]:
    crow(f'(d) Within-subj., all sets, $k={k}$',lambda c,s,k=k: cand[(cand.c==c)&(cand.scale==str(s))&(cand.sets==f'all families, k={k}')].rho.iloc[0])
for k in K:
    crow(f'(e) Pooled (modal file), $k={k}$',lambda c,s,k=k: pool[(pool.c==c)&(pool.scale==s)&(pool.k==k)].rho.iloc[0])
for k in K:
    crow(f'(f) Across subj., $\\Delta L_S$ vs $\\Delta q_1$, $k={k}$',lambda c,s,k=k: xs[(xs.c==c)&(xs.scale==s)&(xs.k==k)]['q1~perron_loading'].iloc[0])
table('Where does ``$L_S$ predicts $q_1$ ($\\rho=0.86$)\'\' come from? Spearman $\\rho(L_S,q_1)$ under different poolings. No saved result equals 0.86 under a single stated pooling (see notes).','tab:S7_LS_q1','l'+'c'*9,
 ['& \\multicolumn{3}{c}{$c=0.01$} & \\multicolumn{3}{c}{$c=0.1$} & \\multicolumn{3}{c}{$c=1$} \\\\','\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}',
  'Pooling '+'& 68 & 114 & 219 '*3+'\\\\'],rows,
 '(a)--(d) computed by us (read-only) from \\texttt{q1/set\\_q1.csv.gz} (all 18 set families incl.\\ random and matched sets; $q_1$ on raw $W$). (e) \\texttt{modal/pooled\\_spearman.csv} (random + ranked + counterfactual sets, pooled over subjects). (f) \\texttt{modal/cycle\\_minus\\_degree\\_node\\_vs\\_set.csv}: across 70 subjects, cycle $-$ degree differences.',size='\\tiny')

open(OUT,'w').write('% S7 set-level tables. Generated from saved results (see S7_notes.txt). Tabular only; captions/notes are in comments above each tabular. Requires booktabs, amsmath.\n\n'+'\n'.join(L))
print('ok')
