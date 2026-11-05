from collections import Counter
import math
import numpy as np
import pytest
from pipe_sim.config import Case
from pipe_sim.geometry import build_model, gouge_depth
from pipe_sim.model import HEX_FACES


@pytest.mark.parametrize('angle',[0,45,90])
@pytest.mark.parametrize('strategy',['structured','adaptive_quads'])
def test_closed_positive_mesh_and_real_gouge(angle,strategy):
    c=Case();c.mesh.strategy=strategy;c.gouge.angle_deg=angle;c.dent.enabled=False;c.mesh.fine_size=6;c.mesh.wall_layers=2
    m=build_model(c)
    assert m.metadata['min_scaled_jacobian']>0.2
    face_counts=Counter(tuple(sorted(ns[i] for i in ii)) for _,ns in m.elements.values() for ii in HEX_FACES.values())
    assert set(face_counts.values())<={1,2}
    exterior=sum(v==1 for v in face_counts.values())
    assert exterior==sum(len(m.surfaces[s]) for s in ['INNER','OUTER','END_MINUS','END_PLUS'])
    inner=np.array([m.nodes[n] for n in m.nsets['INNER_NODES']])
    assert np.allclose(np.linalg.norm(inner[:,:2],axis=1),c.pipe.od/2-c.pipe.thickness)
    outer=np.array([m.nodes[n] for n in m.nsets['OUTER_NODES']])
    radii=np.linalg.norm(outer[:,:2],axis=1)
    assert radii.min()==pytest.approx(c.pipe.od/2-c.gouge.depth)
    assert radii.max()==pytest.approx(c.pipe.od/2)
    # Ensure the surfaces are the intended physical faces, independently of IDs.
    for name,expected in [('INNER',c.pipe.od/2-c.pipe.thickness)]:
        for e,f in m.surfaces[name]:
            nodes=[m.nodes[m.elements[e][1][i]] for i in HEX_FACES[f]]
            assert np.allclose(np.linalg.norm(np.array(nodes)[:,:2],axis=1),expected)


def test_orientation_preserves_intrinsic_profile():
    c=Case();u=np.linspace(-18,18,11);v=np.linspace(-5,5,11)
    expected=gouge_depth(v,u,c.gouge)
    for angle in [45,90]:
        c.gouge.angle_deg=angle;a=math.radians(angle)
        z=u*math.cos(a)-v*math.sin(a);s=u*math.sin(a)+v*math.cos(a)
        assert np.allclose(gouge_depth(s,z,c.gouge),expected)


def test_reject_unresolved_gouge_and_impossible_geometry():
    c=Case();c.gouge.width=1
    with pytest.raises(ValueError,match='fine_size'):c.validate()
    c=Case();c.gouge.depth=c.pipe.thickness
    with pytest.raises(ValueError,match='depth'):c.validate()


def test_finite_support_does_not_fix_pipe_ends(tmp_path):
    from pipe_sim.decks import write_deck
    c=Case();c.gouge.enabled=False;c.dent.enabled=False
    c.mesh.fine_size=6;c.mesh.wall_layers=2
    c.analysis.support='bottom_patch';c.analysis.support_length=80
    m=build_model(c)
    support=m.nsets['BOTTOM_SUPPORT']
    assert len(support)>=3
    assert len(support)<len(m.nsets['BOTTOM'])
    assert all(abs(m.nodes[n][2])<=40+1e-8 and m.nodes[n][1]<0 for n in support)
    assert not set(support).intersection(m.nsets['END_MINUS']+m.nsets['END_PLUS'])
    write_deck(c,m,tmp_path)
    deck=(tmp_path/'job.inp').read_text()
    assert 'BOTTOM_SUPPORT, 2, 2, 0.' in deck
    assert '\nBOTTOM, 2, 2, 0.' not in deck
    c.analysis.support_length=None
    with pytest.raises(ValueError,match='support_length'):c.validate()
