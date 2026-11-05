import numpy as np
from audit_end_surface_traction import shape8, face_vector, face_nodes
from pipe_sim.model import Model


LOCAL = np.array([[-1,-1], [1,-1], [1,1], [-1,1], [0,-1], [1,0], [0,1], [-1,0]])


def test_shape_partition_and_nodal_interpolation():
    for i, (r, s) in enumerate(LOCAL):
        assert np.allclose(shape8(r, s), np.eye(8)[i])
    assert abs(sum(shape8(.31, -.23))-1) < 1e-14


def test_tilted_pressure_resultant_and_translation():
    # x=2r, y=3s, z=r+2s: integral dr cross ds = (-12,-16,24).
    r, s = LOCAL.T
    points = np.column_stack([2*r, 3*s, r+2*s])
    assert np.allclose(face_vector(points), [-12,-16,24], atol=1e-12)
    assert np.allclose(face_vector(points+[7,-8,11]), [-12,-16,24], atol=1e-12)
    assert np.allclose(face_vector(points*1.2), np.array([-12,-16,24])*1.2**2)


def test_quadratic_warp_and_reversed_orientation():
    r, s = LOCAL.T
    points = np.column_stack([r, s, r*r+2*s*s])
    assert np.allclose(face_vector(points), [0,0,4], atol=1e-12)
    assert np.allclose(face_vector(points[[0,3,2,1,7,6,5,4]]), [0,0,-4], atol=1e-12)


def test_hex_face_midside_order():
    model = Model(elements={1: ('C3D20R', list(range(1,21)))})
    assert face_nodes(model, 1, 1) == [1,4,3,2,12,11,10,9]
    assert face_nodes(model, 1, 2) == [5,6,7,8,13,14,15,16]
