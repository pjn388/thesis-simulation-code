"""Read-only feasibility gate for matched native local-intermediate capture.

Does not run/rebuild a solver or infer source-line identity from entry symbols.
"""
import json, subprocess
from pathlib import Path
from audit_plastic_return_map import F, sha256

def main():
    build=F/'sym/vendor/ccx_volume_material_v5_20261019'
    exe=build/'ccx_2.23_material'
    assert sha256(exe)==json.loads((build/'build.json').read_text())['solver_sha256']
    commands=['info address resultsmech_','info address mechmodel_','info address incplas_',
              'info line resultsmech.f:953','info line incplas.f:503',
              'info line incplas.f:747','info sources']
    p=subprocess.run(['gdb','--batch',str(exe)]+[a for c in commands for a in ('-ex',c)],capture_output=True,text=True,timeout=30)
    raw=p.stdout+p.stderr
    debug_missing='compiled without debugging' in raw and 'No symbol table is loaded' in raw
    assert debug_missing, 'Debug capability changed: inspect before choosing capture method'
    full=F/'sym/runs/investigation/volume_material_full_ray_fd_20261019'
    base=json.loads((F/'sym/runs/investigation/volume_material_capture_v5_20261019/assembly.json').read_text())
    frames=[]
    for h in (1e-6,-1e-6):
        path=full/f'h{h}'/'assembly.json';d=json.loads(path.read_text())
        assert d['time']==base['time'] and d['cycle']==base['cycle']==16
        assert d['previous_state']==base['previous_state'] and d['previous_stress']==base['previous_stress']
        assert d['displacement_before']==base['displacement_before']
        assert d['fixed_old_history_unchanged']
        assert d['displacement_imposed']==d['displacement_after']==d['residual']['displacement']
        frames.append(dict(h=h,time=d['time'],cycle=d['cycle'],old_history_identity=True,
                           displacement_identity=True,source=str(path.relative_to(F)),sha256=sha256(path)))
    sources=[Path(__file__),exe,build/'build.json',build/'resultsmech.f',
             F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f',
             F/'results/scripts/capture_late_local_gdb.py',F/'results/scripts/capture_pipe_solver_gdb.py']
    report=dict(kind='matched_local_capture_feasibility',state='blocked_exact_intermediate_alignment',
        capture_authorized=False,native_runs_launched=0,gdb_exit=p.returncode,gdb_evidence=raw,
        retained_frames=frames,selected_ips=[[238,2],[1441,4],[2365,3]],
        finding='Existing function-entry ABI hooks can capture incplas inputs/returns. Existing executable has no source-line/local-variable debug information. These hooks cannot establish resultsmech grad/vkl or incplas scalar iteration/exit identity at the requested source locations.',
        blocker='Exact requested intermediate-call alignment has not been demonstrated. Entry symbols and endpoint stress agreement do not prove it. Per continuation instruction, stop before native replay.',
        next='Prepare an isolated diagnostic build or verified instruction-address map exposing resultsmech/mechmodel/incplas locals, preserving v5. First require parent/disabled coupon identity and retained time/cycle/old-history/displacement identity; then only the matched +/-1e-6 pair. No amplitude sweep. A rebuild is a changed executable and needs independent identity gates before interpreting branch evidence.',
        limitations='No native intermediate capture, branch cause, global derivative acceptance, rollback, convergence benefit or physical burst acceptance. No thesis numerical evidence changed.',
        sources={str(s.relative_to(F)):sha256(s) for s in sources})
    (F/'results/output/matched_local_capture_feasibility_20261019.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('state','native_runs_launched','gdb_exit')}))
if __name__=='__main__':main()
