"""Entry-only x86-64 SysV flag capture on isolated diagnostic processes."""
import gdb
import json
import os
import re
import struct
from pathlib import Path

source = Path(os.environ['THESIS_FLAG_SOURCE']).read_text()
signature = source.split('subroutine e_c3d(', 1)[1].split(')', 1)[0]
names = re.sub(r'\s|&', '', signature).split(',')
records = []
rhs_calls = 0


def value(expr):
    return int(gdb.parse_and_eval(expr))


def read(address, code, count=1):
    size = struct.calcsize(code)*count
    return list(struct.unpack('<'+code*count,
        gdb.selected_inferior().read_memory(address, size).tobytes()))


class Entry(gdb.Breakpoint):
    def stop(self):
        stack = value('$rsp')
        result = {}
        for name in ('nelem', 'nmethod', 'mass', 'stiffness', 'buckling',
                     'rhsi', 'iperturb', 'istep', 'iinc', 'nload'):
            index = names.index(name)+1
            assert index > 6
            pointer = read(stack+8*(index-6), 'Q')[0]
            result[name] = read(pointer, 'i', 2) if name == 'iperturb' else read(pointer, 'i')[0]
        records.append(result)
        return False


class RHS(gdb.Breakpoint):
    def stop(self):
        global rhs_calls
        rhs_calls += 1
        return False


Entry('*e_c3d_', internal=True)
RHS('*e_c3d_rhs_', internal=True)
gdb.execute('run')
Path(os.environ['THESIS_FLAG_OUTPUT']).write_text(json.dumps(
    dict(records=records, rhs_calls=rhs_calls, argument_names=names), indent=2)+'\n')
