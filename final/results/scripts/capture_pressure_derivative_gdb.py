"""Isolated first-assembly probes; imposed diagnostic displacement, then terminate."""
import gdb
import json
import os
import re
import struct
from pathlib import Path

source = Path(os.environ['THESIS_MATRIX_SOURCE']).read_text()
names = re.sub(r'\s|&', '', source.split('subroutine e_c3d(', 1)[1].split(')', 1)[0]).split(',')
records, seen, returns = [], set(), {}


def read(address, count, code):
    return list(struct.unpack('<'+code*count, gdb.selected_inferior().read_memory(
        address, struct.calcsize(code)*count).tobytes()))


def reg(name):
    return int(gdb.parse_and_eval(name))


class Return(gdb.Breakpoint):
    def __init__(self, address):
        super().__init__('*'+hex(address), internal=True)
        self.invocations = {}

    def stop(self):
        record, s, ff = self.invocations.pop(reg('$rsp'), (None, None, None))
        if record is not None:
            record.update(s=read(s, 3600, 'd'), ff=read(ff, 60, 'd'))
            records.append(record)
        return len(records) == 2


class Entry(gdb.Breakpoint):
    def stop(self):
        stack = reg('$rsp')
        def pointer(name):
            index = names.index(name)+1
            assert index > 6
            return read(stack+8*(index-6), 1, 'Q')[0]
        nelem = read(pointer('nelem'), 1, 'i')[0]
        if nelem in seen:
            return False
        seen.add(nelem)
        mi = read(pointer('mi'), 3, 'i')
        imposed = json.loads(os.environ['THESIS_IMPOSED_DISPLACEMENT'])
        for node, values in imposed.items():
            address = pointer('vold')+8*((int(node)-1)*(mi[1]+1)+1)
            gdb.selected_inferior().write_memory(address, struct.pack('<ddd', *values))
        offset = read(pointer('ipkon')+4*(nelem-1), 1, 'i')[0]
        conn = read(reg('$rsi')+4*offset, 20, 'i')
        record = dict(nelem=nelem, connectivity=conn,
            coordinates=[read(reg('$rdi')+24*(n-1), 3, 'd') for n in conn],
            displacements=[read(pointer('vold')+8*((n-1)*(mi[1]+1)+1), 3, 'd') for n in conn],
            initial_stress=read(pointer('sti'), 6*mi[0]*2, 'd'),
            xload=read(pointer('xload'), 4, 'd'), time=read(pointer('time'), 1, 'd')[0],
            flags={name:read(pointer(name), 1, 'i')[0] for name in
                ('nmethod','mass','stiffness','rhsi','buckling','nload','iinc')})
        address = read(stack, 1, 'Q')[0]
        if address not in returns:
            returns[address] = Return(address)
        assert stack+8 not in returns[address].invocations
        returns[address].invocations[stack+8] = (record, pointer('s'), pointer('ff'))
        return False


Entry('*e_c3d_', internal=True)
gdb.execute('run')
assert len(records) == 2
gdb.execute('kill')
Path(os.environ['THESIS_MATRIX_OUTPUT']).write_text(json.dumps(dict(records=records), indent=2)+'\n')
