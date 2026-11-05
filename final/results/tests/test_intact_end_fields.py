from audit_intact_end_fields import complete_fields
from pipe_sim.results import final_frd_displacements


def test_complete_native_fields_exclude_partial_and_keep_component_order():
    lines = ['  100CL  1 0.750000\n', ' -4  STRESS 6 1\n',
             ' -5 SXX\n', ' -5 SYY\n', ' -5 SZZ\n', ' -5 SXY\n',
             ' -5 SYZ\n', ' -5 SZX\n',
             ' -1         1 1.00000E+00-2.00000E+00 3.00000E+00 4.00000E+00 5.00000E+00 6.00000E+00\n',
             ' -3\n', ' -4 DISP 3 1\n', ' -1 1 1 2 3\n']
    fields = list(complete_fields(lines))
    assert fields == [(0.75, 'STRESS', ['SXX','SYY','SZZ','SXY','SYZ','SZX'],
                       {1: [1., -2., 3., 4., 5., 6.]})]


def test_deformation_rejects_later_terminated_unaccepted_frame(tmp_path):
    path = tmp_path/'job.frd'
    path.write_text(' 100CL 1 0.750488\n -4 DISP 3 1\n -1 1 1 2 3\n -3\n'
                    ' 100CL 2 0.750491\n -4 DISP 3 1\n -1 1 4 5 6\n -3\n')
    assert final_frd_displacements(path, [0.7504883]) == {1: [1., 2., 3.]}
    assert final_frd_displacements(path, []) == {}
    assert final_frd_displacements(path) == {1: [4., 5., 6.]}
