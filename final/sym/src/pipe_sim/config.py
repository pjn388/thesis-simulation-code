"""Serializable solver-independent case definitions. All dimensions in mm, N, MPa."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import re


@dataclass
class Pipe:
    od: float = 100.0
    thickness: float = 4.0
    length: float = 400.0


@dataclass
class Gouge:
    enabled: bool = True
    depth: float = 1.0
    length: float = 40.0
    width: float = 12.0
    angle_deg: float = 0.0
    # Smooth raised-cosine profile in unrolled outer-surface coordinates.
    profile: str = "cosine"


@dataclass
class Dent:
    enabled: bool = True
    indenter_diameter: float = 30.0
    travel: float = 3.0
    clearance: float = 0.1
    friction: float = 0.3
    normal_stiffness: float = 2.0e6
    stick_stiffness: float = 2.0e4
    contact_formulation: str = "surface_to_surface"
    # Compression-only face-to-face contact requires the isolated, checked CCX
    # 2.23 TABULAR patch. Preserve legacy LINEAR cases for reproducibility.
    normal_law: str = "linear"


@dataclass
class Material:
    name: str = "PROVISIONAL_STEEL"
    young: float = 209000.0
    poisson: float = 0.3
    # Pairs: true stress [MPa], equivalent plastic logarithmic strain.
    plastic: list[list[float]] = field(default_factory=lambda: [
        [365.0, 0.0], [420.0, 0.02], [480.0, 0.08], [540.0, 0.2],
        [620.0, 0.5], [700.0, 0.8], [750.0, 1.0]])
    source: str = "Demonstration curve only; not a calibrated X52 material."


@dataclass
class Mesh:
    strategy: str = 'structured'
    fine_size: float = 5.0
    far_size: float = 30.0
    wall_layers: int = 3
    order: int = 2
    indenter_size: float = 3.0
    refinement_padding: float = 10.0
    max_elements: int = 150000


@dataclass
class Analysis:
    linear_solver: str = 'default'
    max_pressure: float = 20.0
    failure_peeq: float = 0.8
    initial_increment: float = 0.025
    max_increment: float = 0.1
    # Optional pressure-stage cap; denting/withdrawal retain their own history.
    pressure_max_increment: float | None = None
    # CalculiX face-to-face iteration budget, applied only in pressurisation.
    pressure_contact_iterations: int | None = None
    min_increment: float = 1.0e-6
    max_increments: int = 1000
    output_frequency: int = 1
    threads: int = 2
    timeout_seconds: float = 1800.0
    # Documented fixture idealisations, not inferred experimental fixtures.
    support: str = "bottom_line"
    support_length: float | None = None
    closed_ends: bool = True


@dataclass
class Case:
    name: str = "demo_combined_axial"
    pipe: Pipe = field(default_factory=Pipe)
    gouge: Gouge = field(default_factory=Gouge)
    dent: Dent = field(default_factory=Dent)
    material: Material = field(default_factory=Material)
    mesh: Mesh = field(default_factory=Mesh)
    analysis: Analysis = field(default_factory=Analysis)
    purpose: str = "development"
    # [circumferential arc distance, axial distance] from the crown/pipe centre.
    probe_positions: list[list[float]] = field(default_factory=lambda: [
        [0,0],[0,25],[0,-25],[25,0],[-25,0],[25,25],[-25,-25],[0,50],[50,0]])
    notes: list[str] = field(default_factory=lambda: [
        "Provisional gouge cross-section, material curve and fixture; not validated thesis data.",
        "Gouge exists before indentation. Travel is measured from the nominal outside surface, not residual dent depth.",
        "Indentation is unpressurised; pressure is applied after full indenter withdrawal."])

    def validate(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", self.name):
            raise ValueError("Case name must start with a letter and contain only letters, digits, _ or -")
        p, g, d, m, a = self.pipe, self.gouge, self.dent, self.mesh, self.analysis
        def finite(obj):
            if isinstance(obj, dict):
                for v in obj.values(): finite(v)
            elif isinstance(obj, list):
                for v in obj: finite(v)
            elif isinstance(obj, (int, float)) and not math.isfinite(obj):
                raise ValueError("All numeric parameters must be finite")
        finite(asdict(self))
        if not self.probe_positions or any(len(x)!=2 or abs(x[0])>math.pi*p.od/2 or abs(x[1])>p.length/2 for x in self.probe_positions):
            raise ValueError('Probe positions must be [arc distance, axial distance] inside the pipe domain')
        if not 0 < p.thickness < p.od / 2 or p.length <= 0:
            raise ValueError("Invalid pipe dimensions")
        if g.enabled:
            if not (0 < g.depth < p.thickness and g.length >= g.width > 0):
                raise ValueError("Gouge needs 0 < depth < wall and length >= width > 0")
            if g.profile != "cosine" or not 0 <= g.angle_deg <= 90:
                raise ValueError("Supported gouge: cosine profile, angle from 0 to 90 degrees")
            angle = math.radians(g.angle_deg)
            if g.length * math.cos(angle) + g.width * math.sin(angle) >= p.length:
                raise ValueError("Gouge reaches a pipe end")
            if g.length * math.sin(angle) + g.width * math.cos(angle) >= math.pi * p.od:
                raise ValueError("Gouge wraps around the pipe seam")
        if d.enabled and not (0 < d.travel < p.od / 2 and d.indenter_diameter > 0):
            raise ValueError("Invalid indentation dimensions")
        if d.clearance < 0 or d.friction < 0 or min(d.normal_stiffness, d.stick_stiffness) <= 0:
            raise ValueError("Invalid contact settings")
        if d.contact_formulation not in ('surface_to_surface','node_to_surface'):
            raise ValueError('Unknown contact formulation')
        if d.normal_law not in ('linear','compression_only'):
            raise ValueError('Unknown normal contact law')
        if m.order not in (1, 2) or not isinstance(m.wall_layers, int) or m.wall_layers < 2:
            raise ValueError("Mesh order must be 1 or 2, with at least two wall layers")
        if m.strategy not in ('structured','adaptive_quads'):
            raise ValueError('Unknown mesh strategy')
        if min(m.fine_size, m.far_size, m.indenter_size) <= 0 or m.far_size < m.fine_size:
            raise ValueError("Require 0 < fine_size <= far_size and indenter_size > 0")
        if g.enabled and m.fine_size > g.width / 2:
            raise ValueError("fine_size must resolve the gouge width with at least two elements")
        if m.refinement_padding < 0 or m.max_elements <= 0:
            raise ValueError("Invalid mesh limits")
        if not 0 < a.min_increment <= a.initial_increment <= a.max_increment <= 1:
            raise ValueError("Require 0 < min_increment <= initial_increment <= max_increment <= 1")
        if a.pressure_max_increment is not None and (
                isinstance(a.pressure_max_increment, bool) or
                not a.min_increment <= a.pressure_max_increment <= a.max_increment):
            raise ValueError('pressure_max_increment must lie between min_increment and max_increment')
        if min(a.max_pressure, a.failure_peeq, a.threads, a.timeout_seconds, a.max_increments, a.output_frequency) <= 0:
            raise ValueError("Pressure, failure strain and solver controls must be positive")
        if a.pressure_contact_iterations is not None and (
                type(a.pressure_contact_iterations) is not int or a.pressure_contact_iterations < 1):
            raise ValueError('pressure_contact_iterations must be a positive integer')
        if any(not isinstance(v,int) or isinstance(v,bool) for v in [a.threads,a.max_increments,a.output_frequency,m.max_elements]):
            raise ValueError('Counts and output frequency must be integers')
        if a.support not in ("bottom_line", "bottom_patch", "end_rings"):
            raise ValueError("Unknown support idealisation")
        if a.linear_solver not in ('default','spooles','pardiso'):
            raise ValueError('Unknown linear equation solver')
        if a.support=='bottom_patch' and (a.support_length is None or not 0<a.support_length<=p.length):
            raise ValueError('bottom_patch requires 0 < support_length <= pipe length')
        mat = self.material
        if mat.young <= 0 or not -1 < mat.poisson < 0.5:
            raise ValueError("Invalid elastic constants")
        if mat.plastic:
            if any(len(row) != 2 for row in mat.plastic) or mat.plastic[0][1] != 0:
                raise ValueError("Plastic curve must be [true stress, plastic strain], starting at zero strain")
            for i, (stress, strain) in enumerate(mat.plastic):
                if stress <= 0 or strain < 0 or (i and (strain <= mat.plastic[i-1][1] or stress < mat.plastic[i-1][0])):
                    raise ValueError("Plastic curve must have positive nondecreasing stress and strictly increasing strain")
            if mat.plastic[-1][1] < a.failure_peeq:
                raise ValueError("Plastic curve must extend at least to the failure strain")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key, kind in [("pipe", Pipe), ("gouge", Gouge), ("dent", Dent),
                          ("material", Material), ("mesh", Mesh), ("analysis", Analysis)]:
            if key in data: data[key] = kind(**data[key])
        return cls(**data).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")
