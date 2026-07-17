import csv, glob, json, statistics as st
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=[]
for f in glob.glob('results/per_run/per_run_*.csv'):
    for r in csv.DictReader(open(f)): rows.append(r)
per=defaultdict(list)  # (tok,batch,ds)->list of sbleu per seed
for r in rows: per[(r['tokenizer'],int(r['batch_size']),r['dataset'])].append(float(r['sbleu_mean']))

TOKn={'gpt2':'GPT-2','radbert':'RadBERT','llama2':'LLaMA-2'}
DS=[('discharge','Discharge'),('radiology','Diagnosis'),('mimic_cxr','MIMIC-CXR')]
BATCH=[(64,'#9ecae1'),(128,'#4292c6'),(256,'#08519c')]  # sequential (ordered)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#444',
    'xtick.color':'#444','ytick.color':'#444','axes.labelcolor':'#222','axes.linewidth':0.8})

# ---------- FIGURE 3: S-BLEU box+strip, one subplot per tokenizer ----------
fig,axes=plt.subplots(1,3,figsize=(10,3.8),sharey=True)
for ai,tk in enumerate(['gpt2','radbert','llama2']):
    ax=axes[ai]; ax.grid(True,axis='y',color='#ececec',lw=0.7,zorder=0)
    pos=0; xt=[]; xl=[]
    for dk,dn in DS:
        for bi,(b,col) in enumerate(BATCH):
            vals=per[(tk,b,dk)]
            bp=ax.boxplot([vals],positions=[pos],widths=0.62,patch_artist=True,
                          medianprops=dict(color='#222',lw=1.3),showcaps=False,
                          whiskerprops=dict(color='#888',lw=1),boxprops=dict(lw=0))
            for p in bp['boxes']: p.set_facecolor(col); p.set_alpha(0.85)
            ax.scatter(np.random.default_rng(bi).normal(pos,0.06,len(vals)),vals,
                       s=11,color='#333',alpha=0.7,zorder=4,edgecolors='white',linewidths=0.4)
            pos+=1
        xt.append(pos-2); xl.append(dn); pos+=0.7
    ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_title(TOKn[tk],fontsize=10,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    if ai==0: ax.set_ylabel('S-BLEU')
handles=[plt.Rectangle((0,0),1,1,fc=c) for _,c in BATCH]
fig.legend(handles,[f'Batch {b}' for b,_ in BATCH],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.06))
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig('figures/figure3_sbleu_violin.pdf',bbox_inches='tight')
fig.savefig('figures/figure3_sbleu_violin.png',dpi=200,bbox_inches='tight')
print("saved figure3")

# ---------- FIGURE 4: clinical-entity recall by tokenizer (shows the null) ----------
ent=defaultdict(list)
for f in glob.glob('results/ner2_*_s*.json'):
    p=f.split('/')[-1].replace('.json','').split('_'); ds='_'.join(p[1:-1])
    j=json.load(open(f))
    for t in ['gpt2','radbert','llama2']:
        if t in j: ent[(t,ds)].append(j[t]['recall_pct'])
TOK=[('gpt2','GPT-2','#0072B2','o'),('radbert','RadBERT','#E69F00','s'),('llama2','LLaMA-2','#009E73','^')]
fig,ax=plt.subplots(figsize=(6.2,4.2))
ax.grid(True,axis='y',color='#ececec',lw=0.7,zorder=0)
xpos=np.arange(len(DS)); off=[-0.22,0,0.22]
for (tk,tn,col,mk),o in zip(TOK,off):
    m=[st.mean(ent[(tk,dk)]) for dk,_ in DS]; s=[st.stdev(ent[(tk,dk)]) for dk,_ in DS]
    ax.errorbar(xpos+o,m,yerr=s,fmt=mk,color=col,ms=8,lw=0,elinewidth=1.6,capsize=4,
                mec='white',mew=0.8,label=tn,zorder=3)
ax.set_xticks(xpos); ax.set_xticklabels([d for _,d in DS]); ax.set_ylim(55,90)
ax.set_ylabel('Clinical-entity recall (%)'); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False,loc='lower right',fontsize=9)
ax.set_title('Clinical-entity recall by tokenizer (mean ± SD, 5 seeds)',fontsize=10)
fig.tight_layout()
fig.savefig('figures/figure4_entity_recall.pdf',bbox_inches='tight')
fig.savefig('figures/figure4_entity_recall.png',dpi=200,bbox_inches='tight')
print("saved figure4")
