"""Isolated opt-in native material integration; never changes production files."""
import json, shutil, subprocess, sys
from pathlib import Path
from build_static_linesearch import ROOT,SRC,PARENT,sha
OUT=ROOT/'vendor/ccx_volume_material_v5_20261019'

def main():
    parent=json.loads((PARENT/'build.json').read_text())
    assert sha(PARENT/'ccx_2.23_pardiso')==parent['solver_sha256']
    OUT.mkdir(exist_ok=False)
    original={name:(SRC/name).read_text() for name in ('nonlingeo.c','resultsmech.f')}
    assert sha(SRC/'nonlingeo.c')==parent['compiled_sources']['nonlingeo.c']
    text=original['nonlingeo.c']
    declaration='''extern int thesis_material_enabled(void);
extern void thesis_material_prepare(int,int,int,int,int,int,int,int);
extern void thesis_material_scatter(int,const double*,const double*,const int*,const int*,const char*,const int*,const double*,const double*,int,const int*,double*,double*,int*,int*,int*);
'''
    text=declaration+text
    anchor='NNEW(xstiff,double,(long long)27*mi[0]**ne);'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'\n  thesis_material_prepare(*ne,mi[0],*nmethod,*nmpc,ithermal[0],0,*ntrans,*iprestr);')
    anchor='\tif(nasym==1){\n\t  RENEW(au,double,2*nzs[1]);'
    assert text.count(anchor)==1
    text=text.replace(anchor,'\tif(thesis_material_enabled()) nasym=1;\n'+anchor)
    anchor='\n\tiperturb[0]=iperturb_sav[0];\n\tiperturb[1]=iperturb_sav[1];\n\n      }else{'
    assert text.count(anchor)==1
    text=text.replace(anchor,'\n        if(thesis_material_enabled() && *ne!=ne0){fprintf(stderr,"unsupported added contact elements\\n");exit(86); }\n        thesis_material_scatter(*ne,co,vold,kon,ipkon,lakon,mi,xstateini,xstate,*nstate_,nactdof,ad,au,jq,irow,nzs);\n'+anchor)
    (OUT/'nonlingeo.c').write_text(text)
    text=original['resultsmech.f'];anchor='              xstiff(m1,jj,i)=stiff(m1)\n            enddo'
    assert text.count(anchor)==1
    text=text.replace(anchor,anchor+'''
            call thesis_material_capture(i,jj,kode,emec,elconloc,
     &           plconloc,xstateini(1,jj,i),xstate(1,jj,i),nstate_,
     &           ielas,iorien)''')
    # Restore the source-declared absent kinematic table before native incplas.
    anchor2='     &         xstiff,ncmat_,iperturb)'
    assert text.count(anchor2)==1
    text=text.replace(anchor2,anchor2+"\n          if(kode.eq.-51) call thesis_material_table_scope(\n     &         nplkcon(1,imat),\n     &         plconloc)")
    (OUT/'resultsmech.f').write_text(text)
    for name in ('native_material_runtime.c','native_material_kernel.c'):
        shutil.copy2(ROOT.parent/'results/scripts'/name,OUT/name)
    commands=[];include=Path(sys.prefix)/'include';lib=Path(sys.prefix)/'lib'
    for name in ('nonlingeo','native_material_runtime','native_material_kernel'):
        commands.append(['gcc','-O2','-std=gnu99','-Wall','-Wextra','-Wno-error=return-mismatch','-DPARDISO','-DSPOOLES','-DARPACK','-DMATRIXSTORAGE','-DNETWORKOUT','-DARCH=Linux','-I',str(SRC),'-I',str(ROOT/'vendor/spooles'),'-I',str(include),'-c',str(OUT/(name+'.c')),'-o',str(OUT/(name+'.o'))])
    commands.append(['gfortran','-O2','-cpp','-fallow-argument-mismatch','-I',str(SRC),'-c',str(OUT/'resultsmech.f'),'-o',str(OUT/'resultsmech.o')])
    shutil.copy2(PARENT/'ccx_2.23.a',OUT/'ccx_2.23.a')
    commands.append(['ar','r',str(OUT/'ccx_2.23.a'),*[str(OUT/(n+'.o')) for n in ('nonlingeo','resultsmech','native_material_runtime','native_material_kernel')]])
    commands.append(['gcc','-O2','-o',str(OUT/'ccx_2.23_material'),str(SRC/'ccx_2.23.o'),str(OUT/'ccx_2.23.a'),str(ROOT/'vendor/spooles/spooles.a'),'-l:libarpack.so.2','-L',str(lib),'-Wl,-rpath,'+str(lib),'-l:libmkl_rt.so.2','-l:liblapack.so.3','-l:libblas.so.3','-l:libgfortran.so.5','-l:libquadmath.so.0','-lpthread','-lm','-ldl','-fopenmp'])
    for i,cmd in enumerate(commands):
        with (OUT/f'build_{i}.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    report=dict(verified=False,pressure_authorized=False,commands=commands,parent_solver_sha256=parent['solver_sha256'],solver_sha256=sha(OUT/'ccx_2.23_material'),sources={str(p):sha(p) for p in [Path(__file__),SRC/'nonlingeo.c',SRC/'resultsmech.f',OUT/'nonlingeo.c',OUT/'resultsmech.f',OUT/'native_material_runtime.c',OUT/'native_material_kernel.c']},limitations='Experimental live integration built only. Disabled identity, enabled coupon/state restoration and full-direction sparse FD remain required. No pressure successor or production promotion.')
    (OUT/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(OUT/'ccx_2.23_material')
if __name__=='__main__':main()
