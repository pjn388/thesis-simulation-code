"""Unit conversion and branch-aware descriptive error calculations."""
import numpy as np
import pandas as pd

FACTORS={'mm':('mm',1.),'m':('mm',1000.),'N':('kN',.001),'kN':('kN',1.),
         'bar':('MPa',.1),'MPa':('MPa',1.),'GPa':('MPa',1000.),
         '1':('1',1.),'microstrain':('1',1e-6),'percent':('1',.01),'%':('1',.01)}


def conversion(source, target):
    if source==target:return 1.
    a,fa=FACTORS[source];b,fb=FACTORS[target]
    if a!=b:raise ValueError(f'Incompatible units: {source} -> {target}')
    return fa/fb


def canonical_curve(frame,x_unit,y_unit):
    frame=frame.sort_values('point_order',kind='stable').copy()
    for axis,unit in [('x',x_unit),('y',y_unit)]:
        if frame[axis+'_unit'].nunique()!=1:raise ValueError('Mixed units within a series')
        factor=conversion(frame[axis+'_unit'].iloc[0],unit)
        frame[axis]*=factor
        uncertainty=axis+'_digitization_uncertainty'
        if uncertainty in frame:frame[uncertainty]*=abs(factor)
        frame[axis+'_unit']=unit
    return frame


def loop_branches(frame):
    """Keep traversal order; split the single loading/unloading loop at peak x."""
    frame=frame.reset_index(drop=True)
    peak=int(np.argmax(frame.x.to_numpy()))
    return {'indent':frame.iloc[:peak+1].copy(),'withdraw':frame.iloc[peak:].copy()}


def error_metrics(reference, simulation):
    """Interpolate simulation only within overlap, separately for each branch.

    Digitized point count is not an experimental sample size. Duplicate simulator
    x values on one branch are averaged; no extrapolation or confidence intervals.
    """
    sim=simulation[['x','y']].dropna().groupby('x',as_index=False).y.mean().sort_values('x')
    ref=reference[['x','y']].dropna()
    result={'reference_points':len(ref),'overlap_points':0,'coverage_fraction':0.,
            'rmse':None,'mae':None,'max_absolute_error':None,'normalized_rmse_by_reference_range':None}
    if len(sim)<2:return result,pd.DataFrame()
    # Native solver output rounds coordinates; admit endpoint roundoff only.
    tolerance=max(1e-8,float(sim.x.max()-sim.x.min())*1e-6)
    use=ref[(ref.x>=sim.x.min()-tolerance)&(ref.x<=sim.x.max()+tolerance)].copy()
    result.update(overlap_points=len(use),coverage_fraction=len(use)/len(ref) if len(ref) else 0.,
                  overlap_x_min=float(use.x.min()) if len(use) else None,
                  overlap_x_max=float(use.x.max()) if len(use) else None)
    if len(use)<2:return result,pd.DataFrame()
    use['simulation_y']=np.interp(np.clip(use.x,sim.x.min(),sim.x.max()),sim.x,sim.y)
    use['error']=use.simulation_y-use.y
    rmse=float(np.sqrt(np.mean(use.error**2)));yrange=float(ref.y.max()-ref.y.min())
    result.update(rmse=rmse,mae=float(use.error.abs().mean()),max_absolute_error=float(use.error.abs().max()),
                  normalized_rmse_by_reference_range=rmse/yrange if yrange>0 else None)
    return result,use


def pressure_gauge_curve(probes,node,component,baseline_time=2.):
    """Directional engineering extension relative to the post-dent gauge length.

    E is material Green–Lagrange strain: lambda=sqrt(1+2*E_direction).
    This is still a nearest-IP approximation, not a surface gauge average.
    Spatial log-strain components cannot use this directional conversion.
    """
    selected=probes[probes.requested_node==node].copy()
    if selected.empty:raise ValueError('Requested gauge node missing')
    if set(selected.strain_measure)!= {'green_lagrange'}:
        raise ValueError('Directional gauge conversion requires material Green–Lagrange strain')
    baseline=selected[np.isclose(selected.time,baseline_time)]
    if len(baseline)!=1:raise ValueError('Need one complete zero-pressure post-withdrawal gauge frame')
    press=selected[selected.stage=='pressurise'].sort_values('time').copy()
    ratio=(1+2*press[component])/(1+2*float(baseline[component].iloc[0]))
    if (ratio<=0).any():raise ValueError('Nonphysical gauge stretch')
    press['gauge_engineering_strain']=np.sqrt(ratio)-1
    return press
