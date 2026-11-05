"""Single C3D20R elastic native sparse matrix/residual and fixed-history FD gate."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256

def main(tag):
    root=F/'sym/runs/investigation'/('c3d20_solver_'+tag);root.mkdir(exist_ok=False)
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    xyz=np.vstack([xyz,[(xyz[a]+xyz[b])/2 for a,b in edges]])
    deck='*HEADING\nC3D20R native assembly diagnostic, elastic control\n*NODE\n'+'\n'.join(f'{i+1},'+','.join(map(str,x)) for i,x in enumerate(xyz))
    deck+='\n*ELEMENT,TYPE=C3D20R,ELSET=CUBE\n1,'+','.join(map(str,range(1,16)))+',\n'+','.join(map(str,range(16,21)))+'\n'
    for name,axis,value in [('LEFT',0,0),('RIGHT',0,1),('Y0',1,0),('Z0',2,0)]:
        deck+=f'*NSET,NSET={name}\n'+','.join(str(i+1) for i,x in enumerate(xyz) if x[axis]==value)+'\n'
    deck+='*MATERIAL,NAME=STEEL\n*ELASTIC\n202500.,0.3\n*SOLID SECTION,ELSET=CUBE,MATERIAL=STEEL\n*BOUNDARY\nLEFT,1,1,0\nY0,2,2,0\nZ0,3,3,0\n*STEP,NLGEOM,INC=100\n*STATIC\n0.1,1,1e-8,0.1\n*BOUNDARY\nRIGHT,1,1,0.001\n*END STEP\n'
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    capture=Path(__file__).with_name('capture_element_solver_gdb.py');source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    states={};cases=[]
    for h in (0.,1e-7,-1e-7,5e-8,-5e-8):
        assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
        run=root/f'h{h}';run.mkdir();(run/'job.inp').write_text(deck)
        (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        direction=[] if h==0 else np.eye(base['neq'])[0].tolist()
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE='2',THESIS_DIRECTION=json.dumps(direction),THESIS_AMPLITUDE=str(h),THESIS_OUTPUT=str(run/'assembly.json'),THESIS_ELEMENT_SOURCE=str(source))
        # A baseline needs a direction of sufficient length before neq is known.
        if h==0:env['THESIS_DIRECTION']=json.dumps([0.]*60)
        with (run/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=90)
        assert r.returncode==0,str(run)
        data=json.loads((run/'assembly.json').read_text());states[h]=data
        assert data['nk']==20 and data['mi'][0]==8 and not data['ips']
        assert data['displacement_after']==data['assembly_displacement']==data['displacement_imposed']
        assert data['stress']==data['assembly_stress']
        if h==0:base=data
        else:
            for key in ('previous_state','previous_stress','displacement_before','mapping','time','ttime'):assert data[key]==base[key],key
        r=data['residual'];np.testing.assert_array_equal(r['b'],-np.array(data['internal']));np.testing.assert_array_equal(data['global_matrix']['rhs'],r['b']);assert max(map(abs,r['fext']))==0
        cases.append(dict(run=str(run.relative_to(F)),h=h,hashes={n:sha256(run/n) for n in ('job.inp','capture.gdb','assembly.json','solver.log')}))
    mapping=np.array(base['mapping']).reshape(20,4)[:,1:].ravel();s=np.array(base['element_matrix']).reshape(60,60,order='F');s=np.triu(s)+np.triu(s,1).T
    k=np.zeros((base['neq'],base['neq']))
    for i,a in enumerate(mapping):
        for j,b in enumerate(mapping):
            if a>0 and b>0:k[a-1,b-1]+=s[i,j]
    g=base['global_matrix'];assert g['symmetryflag']==0 and g['inputformat']==0 and g['jq'][0]==1 and g['jq'][-1]==g['nzs']+1
    native=np.diag(g['ad'])
    for col in range(g['neq']):
        for off in range(g['jq'][col]-1,g['jq'][col+1]-1):
            row=g['irow'][off]-1;assert col<row<g['neq'];native[row,col]=native[col,row]=g['au'][off]
    np.testing.assert_allclose(k,native,rtol=2e-14,atol=1e-10)
    rows=[];prev=None
    for h in (1e-7,5e-8):
        fd=(np.array(states[h]['internal'])-states[-h]['internal'])/(2*h);err=float(np.linalg.norm(fd-k[:,0])/np.linalg.norm(fd));assert err<2e-7
        step=None if prev is None else float(np.linalg.norm(fd-prev)/np.linalg.norm(fd));assert step is None or step<2e-7
        rows.append(dict(h=h,relative_error=err,step_change=step));prev=fd
    report=dict(checks_passed=True,neq=base['neq'],cases=cases,comparisons=rows,matrix_max_error=float(np.max(abs(k-native))),provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,source,source.with_name('results.c'),exe)},limitations='One elastic C3D20R element, one free direction, zero external load, no contact/MPC. Deliberately stopped before selected factorization. Does not verify plastic assembly or full pipe. Full pipe matrix must remain sparse.')
    output=F/'results/output'/('c3d20_solver_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['comparisons']))
if __name__=='__main__':main(sys.argv[1])
