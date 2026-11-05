import numpy as np
from audit_elastic_force_balance import integrate, constrained_sum


def test_planar_pressure_vector_both_orders():
    points = [[-2,-3,0],[2,-3,0],[2,3,0],[-2,3,0],
              [0,-3,0],[2,0,0],[0,3,0],[-2,0,0]]
    for order in [2,3]:
        assert np.allclose(integrate(points,order),[0,0,24])


def test_only_unique_constrained_components_are_summed():
    rows = [[1,1,2,300],[2,400,5,6],[3,700,800,900]]
    assert np.allclose(constrained_sum(rows,[(1,1),(1,1),(1,2),(2,3)]),[1,2,6])
