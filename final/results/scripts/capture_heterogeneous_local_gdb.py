"""Native fixed-history local strain probe, selected results cycle and IP."""
import gdb,json,os,struct
from pathlib import Path
records=[];cycle=0;selected=False;finished=False

def read(a,n,c):return list(struct.unpack('<'+c*n,gdb.selected_inferior().read_memory(a,struct.calcsize(c)*n).tobytes()))
def args():
    stack=int(gdb.parse_and_eval('$rsp'))
    return lambda i:int(gdb.parse_and_eval(['$rdi','$rsi','$rdx','$rcx','$r8','$r9'][i-1])) if i<=6 else read(stack+8*(i-6),1,'Q')[0]
class Results(gdb.Breakpoint):
    def stop(self):
        global cycle,selected
        p=args()
        if read(p(36),1,'i')[0]!=-1:return False
        selected=cycle==int(os.environ['THESIS_CYCLE']);cycle+=1
        return False
class Done(gdb.FinishBreakpoint):
    def __init__(self,r,p,target,offset):
        super().__init__(gdb.newest_frame(),internal=True)
        self.r=r;self.target=target;self.p={i:p(i) for i in (3,5,10)};self.offset=offset
    def stop(self):
        global finished
        self.r.update(stress=read(self.p[10],6,'d'),stiffness=read(self.p[5],21,'d'),current=read(self.p[3]+8*self.offset,len(self.r['previous']),'d'))
        records.append(self.r);finished=self.target
        return self.target
class IP(gdb.Breakpoint):
    def stop(self):
        p=args();ns=read(p(21),1,'i')[0];mi=read(p(22),3,'i');ip=read(p(20),1,'i')[0];element=read(p(19),1,'i')[0]
        icmd=read(p(8),1,'i')[0]
        if icmd==3:return False
        offset=ns*((element-1)*mi[0]+ip-1)
        target=selected and ip==int(os.environ['THESIS_IP'])
        before=read(p(6),6,'d');changed=before
        if target:
            direction=json.loads(os.environ['THESIS_DIRECTION']);h=float(os.environ['THESIS_AMPLITUDE'])
            changed=[x+h*y for x,y in zip(before,direction)]
            gdb.selected_inferior().write_memory(p(6),struct.pack('<6d',*changed))
        r=dict(ip=ip,element=element,icmd=icmd,cycle=cycle-1,previous=read(p(4)+8*offset,ns,'d'),strain_before=before,strain=changed,elastic=read(p(1),2,'d'),time=read(p(17),1,'d')[0],ttime=read(p(18),1,'d')[0])
        Done(r,p,target,offset);return False
Results('*results',internal=True);IP('*incplas_',internal=True)
gdb.execute('run');assert finished;gdb.execute('kill')
Path(os.environ['THESIS_OUTPUT']).write_text(json.dumps(dict(records=records),indent=2)+'\n')
