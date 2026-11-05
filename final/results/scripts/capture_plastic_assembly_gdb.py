"""Same-cycle all-IP coupon results and element assembly; diagnostic process only."""
import gdb,json,os,re,struct
from pathlib import Path
records=[]; selected=False; cycle=0; snapshot={}; ips=[]
def read(a,n,c):return list(struct.unpack('<'+c*n,gdb.selected_inferior().read_memory(a,struct.calcsize(c)*n).tobytes()))
def reg(s):return int(gdb.parse_and_eval(s))
def args():
    stack=reg('$rsp')
    return lambda i: reg(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1]) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
class ResultsDone(gdb.FinishBreakpoint):
    def __init__(self,p):
        super().__init__(gdb.newest_frame(),internal=True)
        self.p={i:p(i) for i in (7,10,29,33,34,64,66)}
    def stop(self):
        p=self.p
        snapshot.update(displacement_after=read(p[7],32,'d'),stress=read(p[10],48,'d'),strain=read(p[29],48,'d'),internal=read(p[33],snapshot['neq'],'d'),nodal_internal=read(p[34],32,'d'),current_state=read(p[66],snapshot['state_count'],'d'),ips=ips)
        return False
class Results(gdb.Breakpoint):
    def stop(self):
        global selected,cycle
        p=args()
        if read(p(36),1,'i')[0]!=-1:return False
        this=cycle;cycle+=1
        if this!=int(os.environ['THESIS_CYCLE']):return False
        selected=True
        nk=read(p(2),1,'i')[0];ne=read(p(6),1,'i')[0];mi=read(p(69),3,'i');ns=read(p(73),1,'i')[0]
        assert nk==8 and ne==1 and mi[0]==8
        mapping=read(p(35),32,'i');neq=read(p(51),1,'i')[0]
        before=read(p(7),32,'d');direction=json.loads(os.environ['THESIS_DIRECTION']);h=float(os.environ['THESIS_AMPLITUDE'])
        changed=[u+h*direction[m-1] if m>0 else u for u,m in zip(before,mapping)]
        gdb.selected_inferior().write_memory(p(7),struct.pack('<32d',*changed))
        snapshot.update(cycle=this,time=read(p(57),1,'d')[0],ttime=read(p(58),1,'d')[0],mapping=mapping,neq=neq,displacement_before=before,displacement_imposed=changed,state_count=ns*8,previous_state=read(p(64),ns*8,'d'),previous_stress=read(p(74),48,'d'))
        ResultsDone(p)
        return False
class IPDone(gdb.FinishBreakpoint):
    def __init__(self,r,p):
        super().__init__(gdb.newest_frame(),internal=True);self.r=r;self.p={i:p(i) for i in (3,5,10)}
    def stop(self):
        self.r.update(stress=read(self.p[10],6,'d'),stiffness=read(self.p[5],21,'d'));ips.append(self.r);return False
class IP(gdb.Breakpoint):
    def stop(self):
        if not selected:return False
        p=args();ns=read(p(21),1,'i')[0]
        IPDone(dict(ip=read(p(20),1,'i')[0],icmd=read(p(8),1,'i')[0],previous=read(p(4),ns,'d'),strain=read(p(6),6,'d')),p)
        return False
source=Path(os.environ['THESIS_ELEMENT_SOURCE']).read_text()
names=re.sub(r'\s|&','',source.split('subroutine e_c3d(',1)[1].split(')',1)[0]).split(',')
class ElementDone(gdb.FinishBreakpoint):
    def __init__(self,a):super().__init__(gdb.newest_frame(),internal=True);self.a=a
    def stop(self):snapshot['element_matrix']=read(self.a,3600,'d');return True
class Element(gdb.Breakpoint):
    def stop(self):
        if not selected:return False
        p=args();snapshot['assembly_displacement']=read(p(names.index('vold')+1),32,'d');snapshot['assembly_stress']=read(p(names.index('sti')+1),48,'d')
        ElementDone(p(9));return False
Results('*results',internal=True);IP('*incplas_',internal=True);Element('*e_c3d_',internal=True)
gdb.execute('run')
assert selected and 'element_matrix' in snapshot and len(ips)==8
gdb.execute('kill')
Path(os.environ['THESIS_OUTPUT']).write_text(json.dumps(snapshot,indent=2)+'\n')
