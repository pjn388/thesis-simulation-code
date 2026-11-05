from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


# CalculiX reference node order, also meshio's hexahedron20 convention.
HEX_LOCAL = np.array([
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
    [0,-1,-1],[1,0,-1],[0,1,-1],[-1,0,-1],
    [0,-1,1],[1,0,1],[0,1,1],[-1,0,1],
    [-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]], dtype=float)
HEX_FACES = {1:[0,3,2,1], 2:[4,5,6,7], 3:[0,1,5,4],
             4:[1,2,6,5], 5:[2,3,7,6], 6:[3,0,4,7]}
TET_FACES = {1:[0,2,1], 2:[0,1,3], 3:[1,2,3], 4:[2,0,3]}


@dataclass
class Model:
    nodes: dict[int, list[float]] = field(default_factory=dict)
    elements: dict[int, tuple[str, list[int]]] = field(default_factory=dict)
    nsets: dict[str, list[int]] = field(default_factory=dict)
    elsets: dict[str, list[int]] = field(default_factory=dict)
    surfaces: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, separators=(",", ":")) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d['nodes'] = {int(k): v for k, v in d['nodes'].items()}
        d['elements'] = {int(k): v for k, v in d['elements'].items()}
        return cls(**d)

    def export_vtu(self, path, displacements=None):
        import meshio
        # Reference nodes are not part of the visualisation mesh.
        ids = sorted({n for _, ns in self.elements.values() for n in ns})
        lookup = {n:i for i,n in enumerate(ids)}
        points = np.array([self.nodes[n] for n in ids])
        cells, cell_ids = [], []
        for kind, vtk in [('C3D8','hexahedron'), ('C3D20R','hexahedron20'), ('C3D4','tetra')]:
            group = [(e, ns) for e,(typ,ns) in self.elements.items() if typ == kind]
            if group:
                cells.append((vtk, [[lookup[n] for n in ns] for _,ns in group]))
                cell_ids.append(np.array([e for e,_ in group]))
        pd = {'node_id': np.array(ids)}
        if displacements is not None:
            pd['U'] = np.array([displacements.get(n, [0.,0.,0.]) for n in ids])
        meshio.write_points_cells(path, points, cells, point_data=pd, cell_data={'element_id':cell_ids})
