"""Inject only into an admissible elastic coupon before native stress evaluation.
Reuse the retained matrix/residual capture without changing its original bytes.
"""
import os
from pathlib import Path

base = Path(__file__).with_name('capture_equation_identity_gdb.py').read_text()
injection = r'''
injected = []
class Impose(gdb.Breakpoint):
    def stop(self):
        if injected:
            return False
        stack = reg('$rsp')
        def pointer(index):
            return read(stack+8*(index-6),1,'Q')[0]
        # results.c signature: nk=2, v=7, nactdof=35, iout=36,
        # time=57, mi=69, iinc=105. Read only pointers used below.
        nk = read(reg('$rsi'),1,'i')[0]
        iout = read(pointer(36),1,'i')[0]
        if iout != -1:
            return False
        mapping = read(pointer(35),4*nk,'i')
        assert sorted(m for m in mapping if m > 0) == list(range(1,58))
        direction = json.loads(os.environ['THESIS_DIRECTION'])
        amplitude = float(os.environ['THESIS_AMPLITUDE'])
        base_state = json.loads(os.environ['THESIS_BASE_STATE'])
        displacement = [base_state[m-1]+amplitude*direction[m-1] if m > 0 else 0.0 for m in mapping]
        address = pointer(7)
        assert max(abs(x) for x in read(address,4*nk,'d')) == 0
        gdb.selected_inferior().write_memory(address,struct.pack('<'+'d'*len(displacement),*displacement))
        injected.append(dict(iout=iout, displacement=displacement, mapping=mapping))
        return False
Impose('*results', internal=True)
'''
base = base.replace("Entry('*e_c3d_', internal=True)", injection+"\nEntry('*e_c3d_', internal=True)")
base = base.replace('assert len(records) == 2 and global_record', 'assert len(records) == 2 and global_record and len(injected) == 1')
base = base.replace('residual=residual_record)', 'residual=residual_record, imposed=injected[0])')
exec(compile(base, str(Path(__file__)), 'exec'))
