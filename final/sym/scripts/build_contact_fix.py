"""Build an isolated CCX 2.23 TABULAR face-to-face array correction.

Original source/archive/executable remain untouched. See docs/contact-fix.md.
"""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'vendor/CalculiX/ccx_2.23/src'
OUT=ROOT/'vendor/ccx_tabular_fix'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    metadata={'base_solver_sha256':sha(SOURCE/'ccx_2.23'),'changes':{}}
    patches=[]
    for name in ['springforc_f2f.f','springstiff_f2f.f']:
        original=(SOURCE/name).read_text();modified=original
        for old,new in [('plconloc(82)','plconloc(802)'),('plconloc(81)','plconloc(801)'),('xiso(20)','xiso(200)'),('yiso(20)','yiso(200)')]:
            assert modified.count(old)==1,(name,old)
            modified=modified.replace(old,new)
        (OUT/name).write_text(modified)
        patches.extend(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        metadata['changes'][name]={'original_sha256':sha(SOURCE/name),'modified_sha256':sha(OUT/name)}
        subprocess.run(['gfortran','-O2','-cpp','-fallow-argument-mismatch','-I',str(SOURCE),'-c',name],cwd=OUT,check=True)
    patch=ROOT/'docs/ccx-2.23-tabular-f2f.patch';patch.write_text(''.join(patches))
    shutil.copy2(SOURCE/'ccx_2.23.a',OUT/'ccx_2.23.a')
    subprocess.run(['ar','r','ccx_2.23.a','springforc_f2f.o','springstiff_f2f.o'],cwd=OUT,check=True)
    executable=OUT/'ccx_2.23_tabularfix'
    subprocess.run(['gfortran','-O2','-o',str(executable),str(SOURCE/'ccx_2.23.o'),'ccx_2.23.a',
                    str(ROOT/'vendor/spooles/spooles.a'),'/usr/lib/x86_64-linux-gnu/libarpack.so',
                    '-llapack','-lblas','-lpthread','-lm','-lc','-fopenmp'],cwd=OUT,check=True)
    metadata.update(executable=str(executable),solver_sha256=sha(executable),patch_sha256=sha(patch))
    (OUT/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
