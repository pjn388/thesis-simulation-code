"""CalculiX step-end restart and pressure-continuation input utilities.

The inherited plastic/contact state is read natively, never rebuilt from FRD.
Times are seconds; density is tonne/mm^3 for the mm/N/MPa unit system.
"""
import re
from pathlib import Path
import numpy as np
from .results import HEADER, NUMBER, dat_blocks

DENSITY = 7.85e-9  # 7850 kg/m^3; assumed steel density, not specimen measurement.
ENERGY = '*EL PRINT, ELSET=PIPE, TOTALS=ONLY, FREQUENCY=1\nELSE, ELKE\n'
RESTART = '*RESTART, WRITE, FREQUENCY=1, OVERLAY\n'


def seed_deck(parent):
    """Same two static steps; only density, energy tracking and restart output added."""
    prefix, _ = parent.split('** Stage: pressurise')
    assert prefix.count('*END STEP') == 2
    assert '*DENSITY' not in prefix and '*RESTART' not in prefix
    prefix = prefix.replace('*MATERIAL, NAME=PIPE_STEEL\n',
                            f'*MATERIAL, NAME=PIPE_STEEL\n*DENSITY\n{DENSITY:.12g}\n', 1)
    return prefix.replace('*END STEP', ENERGY + RESTART + '*END STEP')


def ramp_points(duration, count=400):
    """Tabulated quintic ramp: zero endpoint slope/acceleration before interpolation."""
    s = np.linspace(0., 1., count+1)
    return np.column_stack((s*duration, 10*s**3-15*s**4+6*s**5))


def pressure_step(parent, dynamic=False, duration=1., alpha=-.05, target=35.):
    """Preserve follower-pressure faces and proportional end traction.

    For a static restart audit, target=1.75 gives ten .175 MPa increments.
    Dynamic cases start at zero pressure from the withdrawal checkpoint.
    """
    assert duration > 0 and -1/3 <= alpha <= 0 and 0 < target <= 35
    _, stage = parent.split('** Stage: pressurise')
    stage = '** Stage: pressurise'+stage
    original='*STATIC, SOLVER=PARDISO\n0.005, 1., 1e-06, 0.005\n'
    assert stage.count(original)==1
    if dynamic:
        command=f'*DYNAMIC, ALPHA={alpha:.12g}, SOLVER=PARDISO\n{duration*.0005:.12g}, {duration:.12g}, {duration*1e-6:.12g}, {duration*.0025:.12g}\n'
        amplitude='*AMPLITUDE, NAME=PRESSURE_RAMP\n'+''.join(f'{t:.12g}, {a:.12g}\n' for t,a in ramp_points(duration))
        stage=stage.replace('*DLOAD\n','*DLOAD, AMPLITUDE=PRESSURE_RAMP\n',1)
    else:
        assert target==1.75 and duration==1
        command='*STATIC, SOLVER=PARDISO\n0.1, 1., 1e-06, 0.1\n'
        amplitude=''
    stage=stage.replace(original,command,1)
    # Change only magnitudes in the DLOAD block; pressure and end loads scale together.
    lines=stage.splitlines(); active=False
    for i,line in enumerate(lines):
        if line.startswith('*'): active=line.startswith('*DLOAD')
        elif active:
            fields=line.split(','); assert len(fields)==3
            fields[-1]=f' {float(fields[-1])*target/35:.12g}'; lines[i]=','.join(fields)
    stage='\n'.join(lines)+'\n'
    # Retain numerical controls120; output energy from the seed's existing accumulator.
    assert stage.count('*CONTROLS, PARAMETERS=CONTACT\n,,,120\n')==1
    stage=stage.replace('*END STEP',ENERGY+RESTART+'*END STEP')
    # Amplitudes must be defined after RESTART,READ when used in a restarted job.
    return amplitude+stage


def restart_deck(step):
    return '*RESTART, READ\n'+step


def progress_at(metadata, total_time):
    """Native total time includes inherited steps; report only this job's work."""
    if metadata['kind']=='seed':
        index=0 if total_time<=1+1e-8 else 1
        return dict(fraction=min(1.,total_time/2),stage=('indent','withdraw')[index],
                    stage_percent=min(100.,max(0.,total_time-index)*100),pressure=0.)
    fraction=min(1.,max(0.,(total_time-2.)/metadata['duration_seconds']))
    if metadata['kind']=='dynamic':
        points=ramp_points(metadata['duration_seconds'])
        amplitude=float(np.interp(total_time-2.,points[:,0],points[:,1]))
    else: amplitude=fraction
    return dict(fraction=fraction,stage='pressure',stage_percent=100*fraction,
                pressure=metadata['target_pressure_mpa']*amplitude)


def energy_history(path):
    """Read native total energies; tolerate blank lines and signed scalar values."""
    rows={}; pending=None
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            m=HEADER.search(line)
            if m:
                title=line.strip().lower()
                key=('internal_energy_nmm' if title.startswith('total internal energy') else
                     'kinetic_energy_nmm' if title.startswith('total kinetic energy') else None)
                pending=(float(m[2].replace('D','E')),key) if key and m[1].upper()=='PIPE' else None
            elif pending and line.strip():
                if re.fullmatch(r'\s*'+NUMBER+r'\s*',line):
                    time,key=pending;rows.setdefault(time,{'time':time})[key]=float(line.replace('D','E'))
                pending=None
    return rows


def accepted_times(directory):
    times=[]
    for line in (Path(directory)/'job.sta').read_text().splitlines():
        f=line.split()
        if len(f)==7 and f[0].isdigit() and 'U' not in f[2]:
            times.append(float(f[4].replace('D','E')))
    return times


def read_history(directory, crown, ref, pipe_elements, stage, duration=1., target=35.):
    """Small histories only, with STA acceptance and complete PEEQ/U checks.

    This deliberately does not use results.stage_at: restart total time includes
    the two inherited static steps and the pressure ramp is nonlinear in time.
    """
    directory=Path(directory); rows=[]; frame=None; last=None
    accepted=np.asarray(accepted_times(directory))
    def flush():
        if frame is None or frame['count'] != 8*pipe_elements or not frame['u']: return
        if not len(accepted) or np.min(np.abs(accepted-last))>6e-6*max(1.,abs(last)):return
        name=('indent' if last<=1+1e-7 else 'withdraw') if stage=='seed' else 'pressurise'
        if stage=='dynamic':
            ramp=ramp_points(duration);pressure=target*float(np.interp(last-2.,ramp[:,0],ramp[:,1]))
        elif stage=='static_probe':pressure=target*max(0.,min(1.,last-2.))
        else:pressure=0.
        rows.append(dict(time=last,stage=name,pressure_mpa=pressure,max_peeq=frame['peeq'],
            max_peeq_element=frame['element'],max_peeq_ip=frame['ip'],
            crown_inward_displacement_mm=-frame['u'][crown][1],
            indenter_travel_mm=-frame['u'][ref][1],indenter_force_n=-frame['rf'].get(ref,[0,0,0])[1]))
    for kind,group,time,data in dat_blocks(directory/'job.dat'):
        if last is None or abs(time-last)>1e-9:
            flush();last=time;frame={'count':0,'peeq':0.,'element':0,'ip':0,'u':{},'rf':{}}
        if kind=='PEEQ' and group=='PIPE':
            frame['count']+=len(data)
            for e,ip,value in data:
                if value>frame['peeq']: frame.update(peeq=value,element=int(e),ip=int(ip))
        elif kind in ('U','RF'):
            target_rows=frame['u' if kind=='U' else 'rf']
            for r in data:target_rows[int(r[0])]=r[1:4]
    flush()
    energies=energy_history(directory/'job.dat')
    for row in rows:row.update(energies.get(row['time'],{}))
    return rows
