import csv, glob, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

rows=[]
for f in glob.glob('results/per_run/per_run_*.csv'):
    for r in csv.DictReader(open(f)): rows.append(r)

def cell(m):
    d=defaultdict(list)
    for r in rows: d[(r['tokenizer'],int(r['batch_size']),r['dataset'])].append(float(r[m]))
    return d
exact=cell('exact_accuracy_pct'); sb=cell('sbleu_mean'); rg=cell('rougeL_mean')

TOK=[('gpt2','GPT-2','#0072B2','o'),('radbert','RadBERT','#E69F00','s'),('llama2','LLaMA-2','#009E73','^')]
DS=[('discharge','Discharge'),('radiology','Diagnosis'),('mimic_cxr','MIMIC-CXR')]
MET=[(exact,'Exact sentence accuracy (%)',(0,90)),(sb,'S-BLEU',(0.2,0.85)),(rg,'ROUGE-L',(0.2,0.85))]
B=[64,128,256]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
    'axes.edgecolor':'#444','xtick.color':'#444','ytick.color':'#444','axes.labelcolor':'#222'})
fig,axes=plt.subplots(3,3,figsize=(9.2,8.2),sharex=True)
for ri,(dmet,ylab,ylim) in enumerate(MET):
    for ci,(dk,dname) in enumerate(DS):
        ax=axes[ri][ci]
        ax.grid(True,axis='y',color='#e6e6e6',lw=0.7,zorder=0)
        for tk,tname,col,mk in TOK:
            m=[st.mean(dmet[(tk,b,dk)]) for b in B]
            s=[st.stdev(dmet[(tk,b,dk)]) for b in B]
            ax.errorbar(B,m,yerr=s,color=col,marker=mk,ms=6.5,lw=2,capsize=3,
                        mec='white',mew=0.8,zorder=3,label=tname)
        ax.set_xticks(B); ax.set_ylim(*ylim)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ri==0: ax.set_title(dname,fontsize=10,fontweight='bold',pad=6)
        if ci==0: ax.set_ylabel(ylab,fontsize=9)
        if ri==2: ax.set_xlabel('Batch size',fontsize=9)
# single legend top
h,l=axes[0][0].get_legend_handles_labels()
fig.legend(h,l,loc='upper center',ncol=3,frameon=False,fontsize=9.5,bbox_to_anchor=(0.5,1.0))
fig.tight_layout(rect=[0,0,1,0.965])
fig.savefig('figures/figure2_reconstruction_fidelity.pdf',bbox_inches='tight')
fig.savefig('figures/figure2_reconstruction_fidelity.png',dpi=200,bbox_inches='tight')
print("saved figure2 (pdf+png)")
