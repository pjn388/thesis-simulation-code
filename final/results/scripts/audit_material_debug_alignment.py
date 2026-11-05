"""Read-only optimized-local gate after the terminal debug identity capture."""
import json,re,subprocess
from pathlib import Path
from audit_plastic_return_map import F,sha256

def main():
    build=F/'sym/vendor/ccx_volume_material_debug_20261019'
    exe=build/'ccx_2.23_material'
    identity=F/'results/output/material_debug_identity_20261019.json'
    d=json.loads(identity.read_text())
    assert d['state']=='completed' and d['checks_passed']
    assert all(d['retained_identity'].values()) and d['debug_v5_coupon_byte_identity']
    commands=['info line resultsmech.f:958','info scope resultsmech.f:958',
              'info line incplas.f:614','info line incplas.f:762','info scope incplas.f:762']
    p=subprocess.run(['gdb','--batch',str(exe)]+[v for c in commands for v in ('-ex',c)],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    raw=p.stdout+p.stderr
    scopes=raw.split('Scope for ')
    def symbol(scope,name):
        m=re.search(r'Symbol '+name+r' .*?(?=\nSymbol |\Z)',scope,re.S)
        return m.group(0).strip() if m else 'Not present in requested scope'
    result_scope=next(s for s in scopes if s.startswith('resultsmech'))
    inc_scope=next(s for s in scopes if s.startswith('incplas'))
    lines=[line for line in raw.splitlines() if line.startswith('Line ')]
    exit_line=next(line for line in lines if line.startswith('Line 762 '))
    pc=int(re.search(r'address (0x[0-9a-f]+)',exit_line).group(1),16)
    cop=symbol(inc_scope,'cop')
    ranges=[(int(a,16),int(b,16)) for a,b in re.findall(r'Range (0x[0-9a-f]+)-(0x[0-9a-f]+)',cop)]
    cop_at_exit=any(a<=pc<b for a,b in ranges)
    assert not cop_at_exit,'Availability changed; inspect live before deciding'
    out=dict(state='blocked_optimized_exit_locals',baseline_identity_passed=True,
        native_runs_launched=0,matched_pair_authorized=False,physical_validation=False,
        pressure_authorized=False,source_lines=lines,exit_pc=hex(pc),cop_location_at_exit=cop_at_exit,
        resultsmech_locals={n:symbol(result_scope,n) for n in ('vkl','xkl','i','jj')},
        incplas_locals={n:symbol(inc_scope,n) for n in ('cop','dcop','ep','epini','iloop','fu','fiso','dfiso')},
        finding='Terminal debug baseline matches retained v5 at .7875/cycle16 including old history, displacement and residual. Debug scopes expose resultsmech vkl/xkl, but incplas cop has no DWARF location at the mapped post-loop update PC; fu is optimized out. Source-line symbols alone do not supply exact scalar exit capture.',
        blocker='Exact optimized scalar-loop exit/call alignment and live local availability remain unproved. Do not launch matched perturbed pair using source-line guesses.',
        next='Before any native rerun, produce a verified instruction-address/dataflow map for the Newton convergence test and post-loop update, accounting for multiple optimized exits and scalar register lifetimes. If that cannot establish exact read-only capture, retain this blocker; no changed-optimization solver substitution or amplitude sweep.',
        limitations='Static DWARF inspection only, not a live intermediate capture or proof that machine-register recovery is impossible. No branch cause, global native derivative acceptance, rollback or convergence benefit.',
        sources={str(x.relative_to(F)):sha256(x) for x in (Path(__file__),identity,exe,build/'resultsmech.f',build/'incplas.f')})
    path=F/'results/output/material_debug_alignment_20261019.json'
    assert not path.exists(),'Preserve prior report'
    path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('state','baseline_identity_passed','cop_location_at_exit','native_runs_launched')}))
if __name__=='__main__':main()
