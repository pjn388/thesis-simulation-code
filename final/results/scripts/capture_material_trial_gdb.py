"""Replay identical native history, perturb one local strain trial, then stop.

Only a separately named diagnostic process is modified, never production.
"""
import os
from pathlib import Path
base=Path(__file__).with_name('capture_material_tangent_gdb.py').read_text()
base=base.replace('records=[]','records=[]\ntrial_count=0\nselected=False')
base=base.replace('records.append(self.record)\n        return False','records.append(self.record)\n        return selected')
base=base.replace("stack=reg('$rsp')","global trial_count,selected\n        stack=reg('$rsp')")
needle="nstate=read(p(21),1,'i')[0]"
base=base.replace(needle,"""selected=trial_count==int(os.environ['THESIS_TRIAL_INDEX'])
        trial_count+=1
        if selected:
            before=read(p(6),6,'d')
            direction=json.loads(os.environ['THESIS_TRIAL_DIRECTION'])
            amplitude=float(os.environ['THESIS_TRIAL_AMPLITUDE'])
            changed=[v+amplitude*d for v,d in zip(before,direction)]
            gdb.selected_inferior().write_memory(p(6),struct.pack('<6d',*changed))
        """+needle)
base=base.replace("assert records","assert records and selected\ngdb.execute('kill')")
exec(compile(base,str(Path(__file__)),'exec'))
