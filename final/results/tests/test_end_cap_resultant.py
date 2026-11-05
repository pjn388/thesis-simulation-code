import numpy as np
from audit_end_cap_resultant import edge_area


def test_quadratic_green_area_and_translation():
    # x=t, y=t^2 gives integral (x dy-y dx)/2 = integral t^2/2 = 1/3.
    assert abs(edge_area([[-1,1,0], [0,0,0], [1,1,0]])-1/3) < 1e-14
    square = np.array([[0,0,0], [2,0,0], [2,3,0], [0,3,0]])
    def total(points):
        return sum(edge_area([a, (a+b)/2, b]) for a,b in zip(points, np.roll(points,-1,axis=0)))
    assert abs(total(square)-6) < 1e-14
    assert abs(total(square+[10,-7,4])-6) < 1e-13
    assert abs(total(square*1.2)-6*1.2**2) < 1e-13


def test_reference_ratio_invariance_and_differential_expansion():
    inner, outer = 9., 16.
    ratio = inner/(outer-inner)
    for scale in [1., 1.3]:
        assert abs(ratio*((outer-inner)*scale)/(inner*scale)-1) < 1e-14
    # Bore expands while wall projected area is unchanged: original ratio underloads.
    assert ratio*(outer-inner)/(inner*1.2)-1 < 0
