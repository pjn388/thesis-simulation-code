"""Bounded real solver coupons for experimental live integration, no pipe solve."""
import json, os, shutil, subprocess, time
from pathlib import Path
from datetime import datetime,timezone
from audit_plastic_return_map import F,sha256
from pipe_sim.runner import solver_environment
from run_static_linesearch_experiment import fields,comparison
ROOT=F/'sym/runs/investigation/volume_material_integration_v5_20261019'
BUILD=F/'sym/vendor/ccx_volume_material_v5_20261019'
OUT=F/'results/output/volume_material_integration_v5_20261019.json'
def main():
    ROOT.mkdir(exist_ok=False)
    parent=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';exe=BUILD/'ccx_2.23_material'
    report=dict(state='running',checks_passed=False,physical_validation=False,pressure_authorized=False,cases=[],sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),BUILD/'build.json',exe,parent)})
    def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
        source=F/'sym/runs/investigation/c3d20_plastic_v4_20261017T045604Z/c30_h0.0/job.inp'
        text=source.read_text()
        # Original source material and a >200-point native resampling transfer.
        long_source=F/'sym/runs/campaign/orientation_intact_00_adaptive/job.inp'
        material=long_source.read_text().split('*MATERIAL, NAME=PIPE_STEEL\n',1)[1].split('*SOLID SECTION',1)[0]
        long_deck=text.split('*ELASTIC\n',1)[0]+material+'*SOLID SECTION'+text.split('*SOLID SECTION',1)[1]
        report['sources'][str(long_source.relative_to(F))]=sha256(long_source)
        for label,deck in [('material',text),('elastic',text),('resampled_material',long_deck)]:
            if label=='elastic':
                before,after=text.split('*PLASTIC\n',1);deck=before+'*SOLID SECTION'+after.split('*SOLID SECTION',1)[1]
            refs=None
            for variant,binary in [('parent',parent),('disabled',exe),('enabled',exe)]:
                folder=ROOT/(label+'_'+variant);folder.mkdir();(folder/'job.inp').write_text(deck)
                env=solver_environment(1);env.pop('THESIS_VOLUME_MATERIAL',None)
                if variant=='enabled':env['THESIS_VOLUME_MATERIAL']='1'
                start=time.monotonic()
                with (folder/'solver.log').open('w') as log:
                    p=subprocess.run([str(binary),'-i','job'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
                row=dict(label=label,variant=variant,returncode=p.returncode,elapsed=time.monotonic()-start,run=str(folder.relative_to(F)),input_sha256=sha256(folder/'job.inp'))
                report['cases'].append(row);save()
                assert p.returncode==0,'Native coupon failed: '+str(folder)
                actual=fields(folder/'job.dat')
                if variant=='parent':refs=actual
                else:
                    row['comparison']=comparison(refs,actual)
                    assert row['comparison']['passed'],'Unchanged 1e-5 observable gate failed'
                if variant=='disabled':
                    row['dat_byte_identity']=(folder/'job.dat').read_bytes()==(ROOT/(label+'_parent/job.dat')).read_bytes()
                    assert row['dat_byte_identity'],'Disabled parent byte identity failed'
                if variant=='enabled':
                    row['runtime_scatter_observed']='THESIS_VOLUME_MATERIAL scattered' in (folder/'solver.log').read_text()
                    assert row['runtime_scatter_observed'],'No native scatter observed'
                save()
        report.update(state='completed',checks_passed=True)
    except Exception as exc:
        report.update(state='failed',error=str(exc));raise
    finally:
        report.update(finished_utc=datetime.now(timezone.utc).isoformat(),limitations='Native coupon outputs only. No full-ray sparse FD or rollback/state-restoration acceptance. No pressure solve, convergence benefit, physical validation or production promotion.')
        save()
if __name__=='__main__':main()
