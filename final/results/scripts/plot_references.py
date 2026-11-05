"""Plot all recovered paper figures without pretending that every study is matched."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_cases import DATA, RESULTS, digest


def main():
    output=RESULTS/'output/paper_reference';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for source in sorted((DATA/'figures').glob('*/*.csv')):
        data=pd.read_csv(source).fillna({'panel':''});panels=list(data.panel.unique())
        cols=min(3,len(panels));rows=(len(panels)+cols-1)//cols
        fig,axs=plt.subplots(rows,cols,figsize=(6*cols,4.5*rows),squeeze=False,layout='constrained')
        for ax,panel in zip(axs.flat,panels):
            subset=data[data.panel==panel]
            for name,curve in subset.groupby('series',sort=False):
                curve=curve.sort_values('point_order',kind='stable')
                ax.plot(curve.x,curve.y,lw=1.1,marker='.',ms=2.5,label=name)
            ax.set(title=f'Panel {panel}' if panel else source.stem,
                   xlabel=f'{subset.x_quantity.iloc[0]} ({subset.x_unit.iloc[0]})',
                   ylabel=f'{subset.y_quantity.iloc[0]} ({subset.y_unit.iloc[0]})')
            ax.legend(fontsize=6);ax.grid(alpha=.2)
        for ax in list(axs.flat)[len(panels):]:ax.axis('off')
        fig.suptitle(source.parent.name+' / '+source.stem+'\nPublished digitized data only; original units and point order retained',fontsize=13)
        target=output/source.parent.name;target.mkdir(exist_ok=True)
        for ext in ('png','pdf'):fig.savefig(target/(source.stem+'.'+ext),dpi=150)
        plt.close(fig)
        records.append({'source':str(source.relative_to(DATA)),'sha256':digest(source),'rows':len(data),
                        'png':str((target/(source.stem+'.png')).relative_to(RESULTS)),
                        'comparison_status':'paper_reference_only'})
    (output/'index.json').write_text(json.dumps(records,indent=2)+'\n')
    print(f'Rendered {len(records)} paper figures')


if __name__=='__main__':main()
