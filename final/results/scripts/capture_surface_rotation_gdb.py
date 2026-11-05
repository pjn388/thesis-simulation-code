"""Disposable paired element replay; never continues altered flags to assembly."""
import gdb,json,os,re,struct
from pathlib import Path
source=Path(os.environ['THESIS_ELEMENT_SOURCE']).read_text()
names=re.sub(r'\s|&','',source.split('subroutine e_c3d(',1)[1].split(')',1)[0]).split(',')
selected=False;cycle=0;record={};inside=False;shape_records=[]
def read(a,n,c='d'):
    return list(struct.unpack('<'+c*n,gdb.selected_inferior().read_memory(a,struct.calcsize(c)*n).tobytes()))
def args():
    stack=int(gdb.parse_and_eval('$rsp'))
    return lambda i:int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1])) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
def put(a,v,c):gdb.selected_inferior().write_memory(a,struct.pack('<'+c*len(v),*v))
class Results(gdb.Breakpoint):
    def stop(self):
        global cycle,selected
        p=args()
        if read(p(36),1,'i')[0]!=-1:return False
        selected=cycle==2;cycle+=1
        return False
class Done(gdb.FinishBreakpoint):
    def __init__(self,p):
        super().__init__(gdb.newest_frame(),internal=True)
        self.p=p
    def stop(self):
        record.update(matrix=read(self.p('s'),3600),mass_matrix=read(self.p('sm'),3600),force=read(self.p('ff'),60))
        return True
class Element(gdb.Breakpoint):
    def stop(self):
        global inside
        if not selected:return False
        a=args();ptr={n:a(i+1) for i,n in enumerate(names)};p=ptr.__getitem__
        if read(p('nelem'),1,'i')[0]!=int(os.environ['THESIS_ELEMENT']):return False
        mi=read(p('mi'),3,'i');ne=read(p('nelem'),1,'i')[0];offset=read(p('ipkon')+4*(ne-1),1,'i')[0];nodes=read(p('kon')+4*offset,20,'i')
        nload=read(p('nload'),1,'i')[0];loads=read(p('xload'),2*nload)
        ns=read(p('nstate_'),1,'i')[0]
        record.update(argument_names=names,element=ne,nodes=nodes,mi=mi,time=read(p('time'),1),flags_before={n:read(p(n),2 if n=='iperturb' else 1,'i') for n in ('mass','rhsi','stiffness','buckling','iperturb','mortartrafoflag')},loads=loads,load_elements=read(p('nelemload'),2*nload,'i'),sideload=gdb.selected_inferior().read_memory(p('sideload'),20*nload).tobytes().decode('ascii'),coordinates=[read(p('co')+24*(n-1),3) for n in nodes],displacement=[read(p('vold')+8*(mi[1]+1)*(n-1),mi[1]+1) for n in nodes],stress=read(p('sti')+8*6*mi[0]*(ne-1),6*mi[0]),previous_state=read(p('xstateini')+8*ns*mi[0]*(ne-1),ns*mi[0]))
        assert record['flags_before']['mass']==[0] and record['flags_before']['rhsi']==[1]
        assert record['flags_before']['mortartrafoflag']==[0]
        put(p('mass'),[1],'i');put(p('rhsi'),[0],'i')
        if os.environ['THESIS_ZERO_PRESSURE']=='1':put(p('xload'),[0.]*len(loads),'d')
        record['zero_pressure']=os.environ['THESIS_ZERO_PRESSURE']=='1'
        # Manufactured stress-free reference=current whole element. No altered
        # state is allowed to proceed to global assembly or another element.
        rotation=json.loads(os.environ['THESIS_ROTATION'])
        original_coordinates=record['coordinates']
        rotated=[[sum(rotation[i][j]*x[j] for j in range(3)) for i in range(3)] for x in original_coordinates]
        for node,x in zip(nodes,rotated):
            put(p('co')+24*(node-1),x,'d')
            put(p('vold')+8*(mi[1]+1)*(node-1),[0.]*(mi[1]+1),'d')
        for key in ('sti','stx'):put(p(key)+8*6*mi[0]*(ne-1),[0.]*(6*mi[0]),'d')
        for key in ('xstateini','xstate'):put(p(key)+8*ns*mi[0]*(ne-1),[0.]*(ns*mi[0]),'d')
        # Retain stiffness1: e_c3d initializes voldl only with this flag.
        # The loaded/zero pair cancels identical internal contributions.
        put(p('stiffness'),[1],'i')
        record['manufactured']=dict(rotation=rotation,coordinates=rotated,
            displacement=[read(p('vold')+8*(mi[1]+1)*(n-1),mi[1]+1) for n in nodes],
            stress=read(p('sti')+8*6*mi[0]*(ne-1),6*mi[0]),
            previous_state=read(p('xstateini')+8*ns*mi[0]*(ne-1),ns*mi[0]),
            flags={n:read(p(n),1,'i') for n in ('mass','rhsi','stiffness','buckling')})
        inside=True;Done(ptr.__getitem__);self.enabled=False
        return False
class ShapeDone(gdb.FinishBreakpoint):
    def __init__(self, p, local):
        super().__init__(gdb.newest_frame(),internal=True);self.p=p;self.local=local
    def stop(self):
        p=self.p;xs=read(p[4],21);a=xs[:3];b=xs[3:6]
        aa=sum(v*v for v in a);bb=sum(v*v for v in b);ab=sum(v*w for v,w in zip(a,b));det=aa*bb-ab*ab
        assert det>0
        inverse=[[(bb*a[k]-ab*b[k])/det for k in range(3)],[(aa*b[k]-ab*a[k])/det for k in range(3)]]
        shp=read(p[5],56);before=shp[:]
        for j in range(8):
            for k in range(3):shp[7*j+k]=self.local[2*j]*inverse[0][k]+self.local[2*j+1]*inverse[1][k]
        if os.environ['THESIS_METRIC']=='0':shp=before[:]
        put(p[5],shp,'d')
        shape_records.append(dict(xs=xs,local=self.local,before=before,after=shp))
        return False
class Shape(gdb.Breakpoint):
    def stop(self):
        if not inside:return False
        a=args();p=[a(i) for i in range(1,8)]
        if read(p[6],1,'i')[0]!=3:return False
        # Capture local derivatives by an isolated iflag=5 call on entry is unnecessary:
        # use the analytical eight-node serendipity derivatives at the native coordinates.
        r=read(p[0],1)[0];s=read(p[1],1)[0]
        local=[]
        corners=[(-1,-1),(1,-1),(1,1),(-1,1)]
        for u,v in corners:
            local.extend([u*(1+v*s)*(2*u*r+v*s)/4,v*(1+u*r)*(u*r+2*v*s)/4])
        local.extend([-r*(1-s),-(1-r*r)/2,(1-s*s)/2,-s*(1+r),-r*(1+s),(1-r*r)/2,-(1-s*s)/2,-s*(1-r)])
        ShapeDone(p,local);return False
Shape('*shape8q_',internal=True)
Results('*results',internal=True);Element('*e_c3d_',internal=True)
gdb.execute('run');assert 'matrix' in record
assert len(shape_records)==4
record['surface_metric_replacements']=shape_records
gdb.execute('kill');Path(os.environ['THESIS_OUTPUT']).write_text(json.dumps(record)+'\n')
