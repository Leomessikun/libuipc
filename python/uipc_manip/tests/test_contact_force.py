"""Newtons from an exported contact gradient (CPU; the export itself is a GPU feature)."""

import numpy as np
import pytest

from uipc_manip.contact_force import force_summary, vertex_forces

DT = 1.0 / 60.0


class _Instances:
    def __init__(self, index, gradient):
        self._data = {"i": np.asarray(index, dtype=np.int64), "grad": np.asarray(gradient, dtype=np.float64)}

    def size(self):
        return len(self._data["i"])

    def find(self, name):
        return self._data.get(name)


class _Geometry:
    """Stands in for uipc.geometry.Geometry, filled by the feature under test."""

    def __init__(self):
        self._instances = _Instances([], np.zeros((0, 3)))

    def instances(self):
        return self._instances


class _Feature:
    """Replays a recorded export: one (index, gradient) table per primitive type."""

    def __init__(self, tables):
        self.tables = tables

    def contact_primitive_types(self):
        return list(self.tables)

    def contact_gradient(self, prim, geometry):
        index, gradient = self.tables[prim]
        geometry._instances = _Instances(index, gradient)


@pytest.fixture(autouse=True)
def _stub_uipc(monkeypatch):
    """Route the module's lazy uipc imports to the stubs above."""
    import sys
    import types

    uipc = types.ModuleType("uipc")
    uipc.view = lambda attr: attr
    geometry = types.ModuleType("uipc.geometry")
    geometry.Geometry = _Geometry
    monkeypatch.setitem(sys.modules, "uipc", uipc)
    monkeypatch.setitem(sys.modules, "uipc.geometry", geometry)


def test_a_force_is_the_negated_gradient_over_dt_squared():
    # One vertex, one newton upward: the gradient of the incremental potential carries a dt^2.
    feature = _Feature({"PH+N": ([0], [[0.0, -DT * DT, 0.0]])})
    normal, friction = vertex_forces(feature, DT, 1)
    assert normal[0] == pytest.approx([0.0, 1.0, 0.0])
    assert friction[0] == pytest.approx([0.0, 0.0, 0.0])


def test_normal_and_friction_stay_separate():
    feature = _Feature({
        "PT+N": ([0], [[0.0, -2.0 * DT * DT, 0.0]]),
        "PT+F": ([0], [[-0.5 * DT * DT, 0.0, 0.0]]),
    })
    normal, friction = vertex_forces(feature, DT, 1)
    assert normal[0] == pytest.approx([0.0, 2.0, 0.0])
    assert friction[0] == pytest.approx([0.5, 0.0, 0.0])


def test_a_vertex_sums_its_contacts():
    feature = _Feature({"PE+N": ([0, 0, 1], [[0.0, -DT * DT, 0.0]] * 3)})
    normal, _ = vertex_forces(feature, DT, 2)
    assert normal[0] == pytest.approx([0.0, 2.0, 0.0])
    assert normal[1] == pytest.approx([0.0, 1.0, 0.0])


def test_only_the_selected_index_block_is_summed():
    # The arm is the first block and the cloth the second, so a cloth self-contact must not
    # reach the arm's total.
    feature = _Feature({"PP+N": ([0, 5, 9], [[0.0, -DT * DT, 0.0]] * 3)})
    arm, _ = vertex_forces(feature, DT, 3)
    assert arm.sum() == pytest.approx(1.0)  # only vertex 0, pushed up by one newton
    cloth, _ = vertex_forces(feature, DT, 7, first_vertex=3)
    assert cloth[2].sum() == pytest.approx(1.0)  # global 5
    assert cloth[6].sum() == pytest.approx(1.0)  # global 9


def test_a_time_step_change_rescales_nothing_physical():
    # The same physical newton is a four-times smaller gradient at half the step.
    half = DT / 2.0
    coarse = _Feature({"PH+N": ([0], [[0.0, -DT * DT, 0.0]])})
    fine = _Feature({"PH+N": ([0], [[0.0, -half * half, 0.0]])})
    assert vertex_forces(coarse, DT, 1)[0][0][1] == pytest.approx(vertex_forces(fine, half, 1)[0][0][1])


def test_the_summary_separates_the_net_from_what_the_skin_feels():
    # A sleeve gripping an arm: two opposing local forces whose vector sum cancels.
    normal = np.array([[3.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
    friction = np.zeros((2, 3))
    summary = force_summary(normal, friction)
    assert summary["net_normal_n"] == pytest.approx(0.0)
    assert summary["summed_normal_n"] == pytest.approx(6.0)
    assert summary["peak_vertex_normal_n"] == pytest.approx(3.0)
    assert summary["vertices_in_contact"] == 2


def test_an_empty_export_is_zero_not_an_error():
    summary = force_summary(*vertex_forces(_Feature({"PT+N": ([], np.zeros((0, 3)))}), DT, 4))
    assert summary["summed_normal_n"] == 0.0 and summary["peak_vertex_normal_n"] == 0.0
