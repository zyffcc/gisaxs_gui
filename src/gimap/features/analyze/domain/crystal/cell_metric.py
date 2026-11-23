"""Independent entries of the reciprocal metric for a unit cell in its CIF setting."""

from __future__ import annotations

import numpy as np

from .space_groups import space_group, system_of
from .symmetry import parse_operations


def system_for_cell(reference, number=0, symbol="", operations=None):
    """Declared crystallographic system takes precedence over accidental metric symmetry.

    Operations in the CIF (as Operation objects) constrain a symmetric metric through RᵀGR=G.
    A manually entered cell with only a centring is described by its metric.
    """
    if not number and symbol:
        try:
            number = space_group(symbol, cell=reference.as_tuple()).number
        except ValueError:
            pass
    if number:
        return system_of(number)
    a, b, c, alpha, beta, gamma = reference.as_tuple()
    right = np.isclose([alpha, beta, gamma], 90, rtol=0, atol=1e-6)
    if operations:
        if isinstance(operations[0], str):
            operations = parse_operations(operations)
        basis = np.stack([_symmetric(i, j) for i, j in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))])
        constraints = np.concatenate([(op.rotation.T @ basis @ op.rotation - basis).reshape(6, 9).T
                                      for op in operations])
        freedom = 6 - np.linalg.matrix_rank(constraints)
        if freedom != 2:
            return {1: "cubic", 3: "orthorhombic", 4: "monoclinic", 6: "triclinic"}[freedom]
        # For a proper rotation tr(R)=1+2 cos(theta). Remove inversion before testing its order.
        traces = {round(np.linalg.det(op.rotation) * np.trace(op.rotation)) for op in operations}
        if 2 in traces:
            return "hexagonal"
        return "trigonal" if 0 in traces else "tetragonal"
    if right.all():
        if np.isclose(a, b, rtol=1e-7, atol=0):
            return "cubic" if np.isclose(b, c, rtol=1e-7, atol=0) else "tetragonal"
        return "orthorhombic"
    if (np.isclose(a, b, rtol=1e-7, atol=0) and right[:2].all() and abs(gamma - 120) < 1e-6):
        return "hexagonal"
    if np.allclose([a, b], c, rtol=1e-7, atol=0) and np.allclose([alpha, beta], gamma, rtol=0, atol=1e-6):
        return "trigonal"
    return "monoclinic" if right.sum() == 2 else "triclinic"


def _symmetric(i, j):
    matrix = np.zeros((3, 3))
    matrix[i, j] = matrix[j, i] = 1.
    return matrix


class CellMetric:
    """M = (AᵀA)⁻¹ = sum x_j B_j. Then |q(h)|² = (2π)² hᵀMh is linear in the x_j."""

    def __init__(self, reference, system):
        diag = [_symmetric(i, i) for i in range(3)]
        cross = [_symmetric(1, 2), _symmetric(0, 2), _symmetric(0, 1)]
        angles = reference.as_tuple()[3:]
        self.rhombohedral = system == "trigonal" and np.allclose(angles, angles[0], atol=.2, rtol=0)
        if system == "cubic":
            basis, names = [np.eye(3)], ["a"]
        elif self.rhombohedral:
            basis, names = [np.eye(3), sum(cross)], ["a", "c"]
        elif system in ("tetragonal", "hexagonal", "trigonal"):
            basal = diag[0] + diag[1]
            if system != "tetragonal":
                basal += cross[2] / 2
            basis, names = [basal, diag[2]], ["a", "c"]
        elif system == "orthorhombic":
            basis, names = diag, ["a", "b", "c"]
        elif system == "monoclinic":
            # The usual b-unique setting has beta free. Preserve an explicit a/c-unique cell.
            angle = int(np.argmax(np.abs(np.asarray(angles) - 90))) if any(abs(v-90) > 1e-6 for v in angles) else 1
            basis, names = [*diag, cross[angle]], ["a", "b", "c", ("alpha", "beta", "gamma")[angle]]
        elif system == "triclinic":
            basis, names = [*diag, *cross], ["a", "b", "c", "alpha", "beta", "gamma"]
        else:
            raise ValueError("Unknown crystal system.")
        self.basis, self.names = np.asarray(basis), names
        direct = reference.direct_matrix()
        inverse = np.linalg.inv(direct.T @ direct)
        self.start = np.linalg.lstsq(self.basis.reshape(len(basis), 9).T, inverse.ravel(), rcond=None)[0]

    def parameters(self, x, *, native=False):
        """Direct-cell parameters, volume and their exact derivatives with respect to x."""
        metric = np.einsum("n,nij->ij", x, self.basis)
        np.linalg.cholesky(metric)  # refuse an indefinite reciprocal metric even if all observed q² are positive
        g = np.linalg.inv(metric)
        dg = -g @ self.basis @ g
        length = np.sqrt(np.diag(g))
        dl = np.diagonal(dg, axis1=1, axis2=2) / (2 * length)
        values = dict(zip("abc", length))
        derivatives = dict(zip("abc", dl.T))
        for name, i, j in (("alpha", 1, 2), ("beta", 0, 2), ("gamma", 0, 1)):
            cosine = np.clip(g[i, j] / (length[i] * length[j]), -1., 1.)
            values[name] = np.rad2deg(np.arccos(cosine))
            dcos = dg[:, i, j] / (length[i] * length[j]) - cosine * (dl[:, i] / length[i] + dl[:, j] / length[j])
            derivatives[name] = -np.rad2deg(dcos / np.sqrt(1 - cosine**2))
        values["V"] = np.sqrt(np.linalg.det(g))
        derivatives["V"] = -.5 * values["V"] * np.einsum("ij,nji->n", g, self.basis)
        if self.rhombohedral and not native:
            # Conventional hexagonal vectors a_h = b_r - a_r, c_h = a_r + b_r + c_r.
            for name, vector in (("a", np.array([-1, 1, 0])), ("c", np.ones(3))):
                values[name] = np.sqrt(vector @ g @ vector)
                derivatives[name] = np.einsum("i,nij,j->n", vector, dg, vector) / (2 * values[name])
            values["V"] *= 3
            derivatives["V"] *= 3
        report = ["a", "b", "c", "alpha", "beta", "gamma"] if native else self.names + ["V"]
        return {name: float(values[name]) for name in report}, np.array([derivatives[name] for name in report])


def independent_reference_parameters(metric, system, errors):
    """Groups of symmetry-equivalent CIF coordinates with one independent standard uncertainty."""
    # Independent direct-cell parameters; symmetry-tied CIF entries describe the same variable.
    groups = [(0,), (1,), (2,)]
    if system == "cubic" or metric.rhombohedral:
        groups = [(0, 1, 2)]
    elif system in ("tetragonal", "trigonal", "hexagonal"):
        groups = [(0, 1), (2,)]
    if metric.rhombohedral:
        groups.append((3, 4, 5))
    elif system in ("monoclinic", "triclinic"):
        groups.extend((i,) for i, name in enumerate(("alpha", "beta", "gamma"), 3) if name in metric.names)
    for group in groups:
        sigma = next((errors[i] for i in group if i < len(errors) and errors[i] is not None), None)
        if sigma is not None:
            yield group, sigma


def reference_errors(reference, metric, system, errors):
    """Propagate reported CIF s.u.s; equivalent lengths/angles vary together in the CIF setting."""
    if not errors or not any(e is not None for e in errors):
        return {}
    # dV/d(a,b,c,α,β,γ) for the cell in the original CIF setting.
    lengths = np.asarray(reference.as_tuple()[:3])
    angles = np.radians(reference.as_tuple()[3:])
    ca, cb, cg = np.cos(angles)
    v = reference.volume
    dvol = np.r_[v / lengths, np.prod(lengths)**2 / v * np.sin(angles)
                 * (np.array([ca, cb, cg]) - [cb*cg, ca*cg, ca*cb]) * np.pi/180]
    names = ("a", "b", "c", "alpha", "beta", "gamma")
    deriv = {name: np.eye(6)[i] for i, name in enumerate(names) if name in metric.names}
    deriv["V"] = dvol
    if metric.rhombohedral:
        a, angle = reference.a, angles[0]
        basal, axial = np.sqrt(2*(1-ca)), np.sqrt(3*(1+2*ca))
        deriv = {"a": np.array([basal, 0, 0, a*np.sin(angle)/basal*np.pi/180, 0, 0]),
                 "c": np.array([axial, 0, 0, -3*a*np.sin(angle)/axial*np.pi/180, 0, 0]),
                 "V": 3*dvol}
    variances = {}
    for group, sigma in independent_reference_parameters(metric, system, errors):
        for key, gradient in deriv.items():
            partial = gradient[list(group)].sum()
            if partial:
                variances[key] = variances.get(key, 0.) + (partial*sigma)**2
    return {k: float(np.sqrt(v)) for k, v in variances.items()}
