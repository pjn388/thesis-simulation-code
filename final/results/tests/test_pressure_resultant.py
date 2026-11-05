import numpy as np
from pipe_sim.model import Model, HEX_LOCAL
from audit_pressure_resultant import surface_vector, reaction_coverage


def test_curved_closed_hex_normals_cancel():
    points = HEX_LOCAL.copy().astype(float)
    r,s,t = points.T.copy()
    points[:,0] = r + .1*s*t
    points[:,1] = s + .15*r*t
    points[:,2] = t + .2*r*s
    model = Model(nodes={i+1:p.tolist() for i,p in enumerate(points)},
                  elements={1:('C3D20R',list(range(1,21)))},
                  surfaces={'CLOSED':[(1,f) for f in range(1,7)]})
    assert np.linalg.norm(surface_vector(model,'CLOSED',model.nodes)) < 1e-12


def test_coverage_counts_dofs_and_shared_nodes_separately():
    model = Model(nsets={'BOTTOM':[1,2], 'LATERAL_ANCHORS':[1,3],
                        'AXIAL_ANCHOR':[1], 'HISTORY':[1,7]})
    coverage = reaction_coverage(model)
    assert coverage['constrained_dofs'] == 5
    assert coverage['recorded_constrained_dofs'] == [(1,1),(1,2),(1,3)]
    assert coverage['missing_constrained_dofs'] == [(2,2),(3,1)]
    assert not coverage['complete']
