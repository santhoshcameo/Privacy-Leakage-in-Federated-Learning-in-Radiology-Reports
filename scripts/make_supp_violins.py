import csv, glob
from collections import defaultdict
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
rows=[]
for f in glob.glob('results/per_run/per_run_*.csv'):
    for r in csv.DictReader(open(f)): rows.append(r)
TOKn={'gpt2':'GPT-2','radbert':'RadBERT','llama2':'LLaMA-2'}
DS=[('discharge','Discharge'),('radiology','Diagnosis'),('mimic_cxr','MIMIC-CXR')]
BATCH=[(64,'#9ecae1'),(128,'#4292c6'),(256,'#08519c')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#444',
    'xtick.color':'#444','ytick.color':'#444','axes.labelcolor':'#222','axes.linewidth':0.8})
def build(metric, ylabel, fname, ylim):
    per=defaultdict(list)
    for r in rows: per[(r['tokenizer'],int(r['batch_size']),r['dataset'])].append(float(r[metric]))
    fig,axes=plt.subplots(1,3,figsize=(10,3.8),sharey=True)
    for ai,tk in enumerate(['gpt2','radbert','llama2']):
        ax=axes[ai]; ax.grid(True,axis='y',color='#ececec',lw=0.7,zorder=0); pos=0; xt=[]; xl=[]
        for dk,dn in DS:
            for bi,(b,col) in enumerate(BATCH):
                vals=per[(tk,b,dk)]
                bp=ax.boxplot([vals],positions=[pos],widths=0.62,patch_artist=True,
                    medianprops=dict(color='#222',lw=1.3),showcaps=False,
                    whiskerprops=dict(color='#888',lw=1),boxprops=dict(lw=0))
                for p in bp['boxes']: p.set_facecolor(col); p.set_alpha(0.85)
                ax.scatter(np.random.default_rng(bi).normal(pos,0.06,len(vals)),vals,s=11,
                    color='#333',alpha=0.7,zorder=4,edgecolors='white',linewidths=0.4); pos+=1
            xt.append(pos-2); xl.append(dn); pos+=0.7
        ax.set_xticks(xt); ax.set_xticklabels(xl); ax.set_title(TOKn[tk],fontsize=10,fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if ylim: ax.set_ylim(*ylim)
        if ai==0: ax.set_ylabel(ylabel)
    handles=[plt.Rectangle((0,0),1,1,fc=c) for _,c in BATCH]
    fig.legend(handles,[f'Batch {b}' for b,_ in BATCH],loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(0.5,1.06))
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(f'figures/{fname}.pdf',bbox_inches='tight'); fig.savefig(f'figures/{fname}.png',dpi=200,bbox_inches='tight')
    print('saved',fname)
build('exact_accuracy_pct','Exact sentence accuracy (%)','appendix3_exact_accuracy',(15,90))
build('rougeL_mean','ROUGE-L','appendix4_rougeL',(0.2,0.85))
