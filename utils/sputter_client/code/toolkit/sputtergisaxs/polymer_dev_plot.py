'All selected frame evolution; source CSV and conditional intervals explicit.'
from pathlib import Path
import csv,sys,json
import numpy as np


def plot(rows,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out)
    if rows is None:
        with (out/'frames.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    def values(key):
        def num(r):
            try:return float(r.get(key,np.nan))
            except (ValueError,TypeError):return np.nan
        return np.asarray([num(r) for r in rows])
    t=values('time_s');clock=np.isfinite(t).all()
    if not clock:t=np.arange(len(rows))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.labelsize':8,'axes.titlesize':8,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.6})
    fig,axes=plt.subplots(3,2,figsize=(7.2047244,6.6929134),sharex=True)
    fig.subplots_adjust(left=.11,right=.98,bottom=.10,top=.84,wspace=.42,hspace=.46)
    a=axes.ravel();colors=['#0072B2','#D55E00','#009E73']
    for ax,name in zip(a[:2],['D_nm','R_number_mean_nm']):
        for m in (3,2,1):
            y=values(f'mode{m}_'+name);ax.plot(t,y,color=colors[m-1],lw=.45,alpha=.75,label=f'Mode {m}',rasterized=True)
        lo=values('mode1_'+name+'_lo68');hi=values('mode1_'+name+'_hi68')
        ax.fill_between(t,lo,hi,color=colors[0],alpha=.13,lw=0,rasterized=True,label='Mode 1: 68% conditional')
        refined=values('refined_'+name);ok=np.isfinite(refined)
        ax.scatter(t[ok],refined[ok],s=3,marker='.',color='#111111',label='Budgeted refinement',rasterized=True)
        positive=np.r_[lo[np.isfinite(lo)&(lo>0)],hi[np.isfinite(hi)&(hi>0)]]
        if len(positive) and positive.max()/positive.min()>30:ax.set_yscale('log')
    dc=values('D_corr_nm');a[0].scatter(t,dc,s=2,color='#555555',label='Direct 2π/q*',rasterized=True)
    a[0].set_ylabel('Spacing D (nm)');a[1].set_ylabel('Radius R (nm)')
    for ax,key,label in [(a[2],'FWHM_nm_inv','Peak FWHM (nm⁻¹)'),(a[3],'Yoneda_shift_deg','Yoneda shift (deg)'),(a[4],'highq_log_slope','High-q log slope')]:
        ax.plot(t,values(key),color='#444444',lw=.55,rasterized=True);ax.set_ylabel(label)
    stage_names=['pre_nucleation_candidate','separate_islands_candidate','coalescence_candidate','continuous_film_candidate']
    stage=np.array([stage_names.index(r['stage']) if r['stage'] in stage_names else -1 for r in rows])
    a[5].scatter(t,stage,s=2,color='#6A3D9A',rasterized=True);a[5].set_yticks([-1,0,1,2,3],['Unresolved','Pre-nucl.','Islands','Coalesc.','Film']);a[5].set_ylabel('Conditional stage')
    for j,ax in enumerate(a):
        ax.set_title('abcdef'[j],loc='left',fontweight='bold',pad=5)
        ax.tick_params(length=2,width=.6);ax.margins(x=.01)
    for ax in a[-2:]:ax.set_xlabel('Header elapsed time (s)' if clock else 'Selected frame order (header clock incomplete)')
    handles,labels=a[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.53,.94),frameon=False,columnspacing=1.5)
    fig.text(.11,.975,out.parent.name+' / '+out.name+' | '+str(len(rows))+' selected frames',fontsize=8,va='top')
    fig.text(.11,.035,'Polymer development v0 | Conditional network ranges; domain gate pending; no independent structural validation.',fontsize=6)
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    if str(scripts) not in sys.path:sys.path.insert(0,str(scripts))
    from figure_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig,json_out=str(out/'evolution.alignment.json'),tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    fig.savefig(out/'evolution.pdf',dpi=300)
    fig.savefig(out/'evolution.svg',dpi=300)
    fig.savefig(out/'evolution.png',dpi=600)
    plt.close(fig)
    (out/'evolution.source.json').write_text(json.dumps(dict(source='frames.csv',rows=len(rows),excluded_rows=0,
        no_peak_D='Not plotted; explicit reason in CSV',interval='Mode1 raw conditional68%; all3 modes68/95% in CSV',
        stage='Observation-conditioned candidate, not morphology truth'),indent=2)+'\n',encoding='utf-8')
