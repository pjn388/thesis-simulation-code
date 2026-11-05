import numpy as np
from audit_nodal_pressure_loads import nodal_loads
from audit_end_surface_traction import face_vector

POINTS = np.array([[-2,-3,0],[2,-3,0],[2,3,0],[-2,3,0],
                   [0,-3,0],[2,0,0],[0,3,0],[-2,0,0]],dtype=float)


def test_quadratic_consistent_loads_have_negative_corner_forces():
    for order in [2,3]:
        loads=nodal_loads(POINTS,2,order)
        assert np.allclose(loads[:4,2],-4)
        assert np.allclose(loads[4:,2],16)
        assert np.allclose(loads[:,:2],0)
        assert np.allclose(loads.sum(axis=0),[0,0,48])


def test_warped_resultant_translation_and_traction_sign():
    p=POINTS.copy()
    p[:,2]=[0,1,3,-1,0.4,2,1,-0.5]
    loads=nodal_loads(p,3,3)
    assert np.allclose(loads.sum(axis=0),3*face_vector(p))
    assert np.allclose(nodal_loads(p+[11,22,33],3,3),loads)
    assert np.allclose(nodal_loads(p,-3,3),-loads)


def test_pressure_loads_preserve_moment_on_planar_tilted_face():
    p=POINTS.copy()
    p[:,2]=0.3*p[:,0]+0.4*p[:,1]+5
    loads=nodal_loads(p,2)
    assert np.allclose(np.cross(p,loads).sum(axis=0),np.cross(p[:4].mean(axis=0),loads.sum(axis=0)))
