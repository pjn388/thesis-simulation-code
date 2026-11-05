"""Plot the published X52 curve and the verified deck input; never run a solver."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OUT = BASE / 'evidence/material_figure_20261026'
FIGURES = BASE / 'figures'
E, SY = 208000.0, 365.0
BLUE, ORANGE, RED, GREY = '#173d59', '#b95922', '#a63437', '#667580'


def main():
    source = ROOT / 'final/data/figures/zhao2021/fig11_material_curve.csv'
    case_path = ROOT / ('final/sym/runs/campaign/'
        'full_zhao2021_p1_adaptive_pressure005_20261020/case.json')
    deck_path = case_path.with_name('job.inp')
    source_rows = list(csv.DictReader(source.open()))
    raw = np.array([[float(r['x']), float(r['y'])] for r in source_rows])
    material = json.loads(case_path.read_text())['material']
    assert material['young'] == E and material['poisson'] == 0.3
    table = np.array(material['plastic'])  # stress, plastic strain
    selected, omitted = [], []
    previous = [SY, 0.0]
    for row, (total, stress) in zip(source_rows, raw):
        plastic = total - stress / E
        if plastic > previous[1] and stress >= previous[0]:
            selected.append([stress, plastic])
            previous = [stress, plastic]
        else:
            omitted.append(int(row['point_order']))
    reconstructed = np.array([[SY, 0.0]] + selected + [[previous[0], 0.8]])
    assert np.array_equal(table, reconstructed)
    assert (len(raw), len(selected), omitted) == (272, 270, [209, 239])
    lines = deck_path.read_text().splitlines()
    start = next(i for i, s in enumerate(lines) if s.upper().startswith('*PLASTIC'))
    deck = []
    for line in lines[start + 1:]:
        if line.startswith('*'):
            break
        if line.strip():
            deck.append([float(v) for v in line.split(',')[:2]])
    assert np.shape(deck) == table.shape
    deck_error = float(np.max(np.abs(np.array(deck) - table)))
    assert deck_error <= 5.1e-13

    # Keep the two strain conventions separate. Restore elastic strain only
    # for panel (a), which compares source and adopted input on identical axes.
    total_input = table[:, 1] + table[:, 0] / E
    source_plastic = raw[:, 0] - raw[:, 1] / E
    supported = table[-2, 1]
    equation_stress = np.linspace(SY, table[-1, 0], 400)
    equation_total = equation_stress / E + SY / E * 0.002 * (equation_stress / SY)**9.98
    assert np.allclose(total_input[1:-1],
        np.array(selected)[:, 1] + np.array(selected)[:, 0] / E, rtol=0, atol=0)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.labelsize': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'savefig.dpi': 240})
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.6))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.09, top=0.95,
                        wspace=0.30, hspace=0.50)
    a, b, c, d = axes.flat
    for ax in axes.flat:
        ax.grid(alpha=0.17, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel('True stress (MPa)')
        ax.tick_params(labelsize=8)

    a.plot(total_input, table[:, 0], color=BLUE, lw=1.9,
           label='Adopted input: total strain restored')
    a.plot(raw[:, 0], raw[:, 1], color=GREY, lw=0.6, alpha=0.75,
           marker='o', ms=3.2, mfc='white', mec=GREY, markevery=14,
           label='Digitised Zhao Figure 11', zorder=3)
    a.plot(equation_total, equation_stress, '--', color=RED, lw=1.6,
           label='Equation as printed (not used)')
    a.plot(total_input[-2:], table[-2:, 0], color=ORANGE, lw=2.8)
    a.set(title='(a) Source and adopted representation', xlim=(0, 0.83),
          ylim=(340, 705), xlabel='Total true strain (dimensionless)')
    a.legend(loc='lower right', fontsize=7, frameon=False, handlelength=2.4)
    a.annotate('Figure-based input\nfollows the published curve',
               xy=(0.43, 624), xytext=(0.25, 460), fontsize=8, color=BLUE,
               arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))

    b.plot(table[:-1, 1], table[:-1, 0], color=BLUE, lw=2,
           label='Adopted deck hardening table')
    b.plot(table[-2:, 1], table[-2:, 0], color=ORANGE, lw=3,
           label='Added constant-stress tail')
    b.scatter([0], [SY], marker='s', s=23, color=BLUE, zorder=4,
              label='Imposed yield anchor')
    b.set(title='(b) Adopted hardening input', xlim=(-0.012, 0.83),
          ylim=(340, 705), xlabel='Plastic strain (dimensionless)')
    b.legend(loc='lower right', fontsize=7, frameon=False)
    b.text(0.04, 0.93, r'$E=208$ GPa; $\nu=0.3$', transform=b.transAxes,
           fontsize=8, va='top')

    c.plot(table[:-1, 1], table[:-1, 0], color=BLUE, lw=1.8)
    c.scatter(source_plastic[:8], raw[:8, 1], s=19, facecolors='white',
              edgecolors=GREY, linewidths=0.9, zorder=3)
    c.scatter([0], [SY], marker='s', s=28, color=BLUE, zorder=4)
    c.annotate('365 MPa anchor\n(imposed)', xy=(0, SY), xytext=(0.009, 357),
               fontsize=8, arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
    c.annotate('First recovered point\n430.8 MPa',
               xy=(source_plastic[0], raw[0, 1]), xytext=(0.001, 461),
               fontsize=8, arrowprops=dict(arrowstyle='->', color=GREY, lw=0.8))
    c.set(title='(c) Near-yield detail', xlim=(-0.001, 0.026),
          ylim=(345, 487), xlabel='Plastic strain (dimensionless)')
    c.set_xticks([0, 0.01, 0.02])

    d.axvspan(supported, 0.8, color=ORANGE, alpha=0.10)
    d.axvline(supported, color=GREY, ls=':', lw=1.1)
    d.axvline(0.8, color=RED, ls='--', lw=1.1)
    d.plot(table[:-1, 1], table[:-1, 0], color=BLUE, lw=1.8)
    d.scatter(source_plastic[-9:], raw[-9:, 1], s=19, facecolors='white',
              edgecolors=GREY, linewidths=0.9, zorder=3)
    d.plot(table[-2:, 1], table[-2:, 0], color=ORANGE, lw=3, zorder=4)
    d.text(supported - 0.001, 665.7, 'Recovered end\n0.7910574',
           ha='right', va='top', fontsize=7.5, color=GREY)
    d.text(0.7994, 665.7, 'Criterion\n0.8', ha='right', va='top',
           fontsize=7.5, color=RED)
    d.text((supported + 0.8)/2, 659.5, 'Added tail', ha='center',
           fontsize=8, color=ORANGE)
    d.set(title='(d) Terminal extension',
          xlim=(0.778, 0.803), ylim=(658.5, 666.2),
          xlabel='Plastic strain (dimensionless)')
    d.set_xticks([0.78, 0.79, 0.8])
    d.set_yticks([660, 663, 666])

    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    for extension in ['pdf', 'png']:
        metadata = {'Creator': 'X52 material provenance figure'}
        if extension == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(FIGURES / f'zhao_material_input.{extension}', metadata=metadata)
    plt.close(fig)

    with (OUT / 'plotted-data.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['series', 'order', 'total_true_strain', 'plastic_strain',
                         'true_stress_MPa', 'origin'])
        for i, ((total, stress), plastic) in enumerate(zip(raw, source_plastic), 1):
            writer.writerow(['digitised_figure11', i, total, plastic, stress,
                             'published model curve; not raw tensile records'])
        for i, ((stress, plastic), total) in enumerate(zip(table, total_input), 1):
            origin = 'imposed yield anchor' if i == 1 else (
                'added constant-stress endpoint' if i == len(table) else 'retained source point')
            writer.writerow(['adopted_deck_input', i, total, plastic, stress, origin])
        for i, (total, stress) in enumerate(zip(equation_total, equation_stress), 1):
            writer.writerow(['printed_equation_not_used', i, total, total-stress/E,
                             stress, 'source consistency comparison only'])
    provenance = [source, case_path, deck_path, Path(__file__).resolve(),
        FIGURES/'zhao_material_input.pdf', FIGURES/'zhao_material_input.png',
        OUT/'plotted-data.csv']
    report = dict(source_points=len(raw), retained_source_points=len(selected),
        omitted_point_orders=omitted, input_rows=len(table), case_exact_match=True,
        deck_max_absolute_error=deck_error, recovered_plastic_extent=float(supported),
        input_last_plastic_strain=float(table[-1, 1]),
        restored_total_strain_used_only_for_source_comparison=True,
        scope='Published source and verified supplied input, not simulated coupon output or native resampling.',
        rationale='Traceable to the validation study, follows its plotted response and is shared across the matched family.',
        limitations='Yield anchor and tail are assumptions; physical material validity and material-choice sensitivity remain unresolved.',
        provenance=[dict(path=str(p.relative_to(ROOT)),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in provenance])
    (OUT/'figure-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Created zhao_material_input.pdf/.png; source selection, case table and deck checks passed.')


if __name__ == '__main__':
    main()
