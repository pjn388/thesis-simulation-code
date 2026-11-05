"""Add DWARF at unchanged O2 in an isolated archive; gate on object code identity."""
import json, shutil, subprocess
from pathlib import Path
from build_static_linesearch import ROOT, SRC, sha

OUT = ROOT/'vendor/ccx_volume_material_debug_20261019'
PARENT = ROOT/'vendor/ccx_volume_material_v5_20261019'

def main():
    meta = json.loads((PARENT/'build.json').read_text())
    assert sha(PARENT/'ccx_2.23_material') == meta['solver_sha256']
    OUT.mkdir(exist_ok=False)
    shutil.copy2(PARENT/'ccx_2.23.a', OUT/'ccx_2.23.a')
    report = dict(state='building', pressure_authorized=False, objects=[], commands=[],
                  parent_solver_sha256=meta['solver_sha256'], sources={})
    def run(cmd):
        report['commands'].append(cmd)
        with (OUT/f'command_{len(report["commands"])}.log').open('w') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    try:
        for name in ('resultsmech', 'mechmodel', 'incplas'):
            source = (PARENT if name == 'resultsmech' else SRC)/(name+'.f')
            target = OUT/(name+'.f'); shutil.copy2(source, target)
            report['sources'][str(source)] = sha(source)
            run(['gfortran','-O2','-g','-cpp','-fallow-argument-mismatch','-I',str(SRC),'-c',str(target),'-o',str(OUT/(name+'.o'))])
            old = OUT/(name+'_parent.o')
            old.write_bytes(subprocess.check_output(['ar','p',str(PARENT/'ccx_2.23.a'),name+'.o']))
            for label, obj in [('parent',old),('debug',OUT/(name+'.o'))]:
                run(['objcopy','-O','binary','--only-section=.text',str(obj),str(OUT/(name+'_'+label+'.text'))])
            identical = (OUT/(name+'_parent.text')).read_bytes() == (OUT/(name+'_debug.text')).read_bytes()
            report['objects'].append(dict(name=name,text_byte_identity=identical,object_sha256=sha(OUT/(name+'.o'))))
            assert identical, name+' machine code changed; stop before native use'
            run(['ar','r',str(OUT/'ccx_2.23.a'),str(OUT/(name+'.o'))])
        link = [str(OUT)+v[len(str(PARENT)):] if v.startswith(str(PARENT)) else v for v in meta['commands'][-1]]
        run(link)
        exe = OUT/'ccx_2.23_material'
        commands = ['info line resultsmech.f:953','info line incplas.f:503','info line incplas.f:747','info scope incplas.f:503','info scope resultsmech.f:953']
        p = subprocess.run(['gdb','--batch',str(exe)]+[v for c in commands for v in ('-ex',c)],capture_output=True,text=True,timeout=30)
        (OUT/'debug_symbols.txt').write_text(p.stdout+p.stderr)
        assert p.returncode == 0 and 'No symbol table' not in p.stdout+p.stderr
        report.update(state='built_code_identity_passed',solver_sha256=sha(exe),debug_evidence=p.stdout+p.stderr,
                      next='Require fresh parent/disabled/enabled coupon identity, then retained time/cycle/history/displacement alignment before only matched +/-1e-6 intermediate capture.',
                      limitations='Object text identity and source symbols only; optimized locals may be unavailable. No native replay, derivative acceptance, rollback or pressure authorization.')
    except Exception as exc:
        report.update(state='failed',error=str(exc)); raise
    finally:
        report['sources'][str(Path(__file__))] = sha(Path(__file__))
        (OUT/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        (ROOT.parent/'results/output/material_debug_build_20261019.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()
