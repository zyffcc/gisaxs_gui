"""Where a reflection of a fibre-textured film appears: refraction and the two DWBA channels.

Inside a film of critical angle α_c the incident wave has the vertical wavevector

    k_z,i = k Re√(sin²α_i − sin²α_c)

A reciprocal-lattice point at (q_∥, q_z) (q_z inside the film, the true one) is seen along two exit directions
(the distorted-wave Born approximation of a film on a reflecting substrate; Jiang, 2015, GIXSGUI; Toney &
Brennan, 1989):

* **transmitted** channel (T): the direct incident wave is scattered upwards, k_z,f = q_z − k_z,i;
* **reflected** channel (R): the wave reflected by the substrate is scattered upwards, k_z,f = q_z + k_z,i.

The two terms with a reflected exit wave leave along the same two directions. A wave with k_z,f ≥ 0 leaves the film
at the vacuum exit angle

    sin α_f = √( (k_z,f / k)² + sin²α_c )

and the q map of Analyze places it at the q_z of that exit angle: k (sin α_f + sin α_i) for the kinematic map, or
the refracted q_z of (2.18) when the map is computed for a film of critical angle α_c,map. With α_c,map = α_c the
T channel falls on the true q_z. q_∥ is the same in the film and in vacuum. A T spot with q_z < k_z,i is scattered
into the substrate and not seen.

The flat detector does not see q_∥ < k |cos α_f − cos α_i| (the missing wedge, ``reachable``).
"""

from __future__ import annotations

import math

import numpy as np

TRANSMITTED, REFLECTED = "T", "R"
CHANNELS = (TRANSMITTED, REFLECTED)


def _sin2(degrees: float) -> float:
    return math.sin(math.radians(float(degrees))) ** 2


def incident_kz(k: float, incidence_deg: float, critical_deg: float) -> float:
    """k_z,i = k Re√(sin²α_i − sin²α_c) (Å⁻¹); 0 below the critical angle."""
    return k * math.sqrt(max(_sin2(incidence_deg) - _sin2(critical_deg), 0.0))


def exit_sine(qz, k: float, incidence_deg: float, critical_deg: float, channel: str) -> np.ndarray:
    """sin α_f in vacuum of each spot of true ``qz`` seen in ``channel`` (NaN where it does not leave the film)."""
    qz = np.asarray(qz, dtype=float)
    kzi = incident_kz(k, incidence_deg, critical_deg)
    kzf = qz - kzi if channel == TRANSMITTED else qz + kzi
    with np.errstate(invalid="ignore"):
        sine = np.sqrt((kzf / k) ** 2 + _sin2(critical_deg))
    return np.where((kzf >= 0.0) & (sine <= 1.0), sine, np.nan)


def map_qz(sine, k: float, incidence_deg: float, map_critical_deg: float = 0.0) -> np.ndarray:
    """q_z on the q map of the rays leaving at sin α_f: kinematic, or refracted for α_c,map > 0 (NaN below it)."""
    sine = np.asarray(sine, dtype=float)
    if map_critical_deg > 0.0:
        critical = _sin2(map_critical_deg)
        with np.errstate(invalid="ignore"):
            exit_term = np.where(sine * sine >= critical, np.sqrt(np.maximum(sine * sine - critical, 0.0)), np.nan)
        return k * (exit_term + math.sqrt(max(_sin2(incidence_deg) - critical, 0.0)))
    return k * (sine + math.sin(math.radians(incidence_deg)))


def observed_qz(qz, k: float, incidence_deg: float, critical_deg: float, map_critical_deg: float,
                channel: str) -> np.ndarray:
    """The q_z at which the spot of true ``qz`` appears on the q map in ``channel``."""
    return map_qz(exit_sine(qz, k, incidence_deg, critical_deg, channel), k, incidence_deg, map_critical_deg)


def reachable(q_parallel, sine, k: float, incidence_deg: float) -> np.ndarray:
    """Whether a flat detector sees in-plane magnitude ``q_parallel`` at exit sine ``sine`` (outside the wedge)."""
    sine = np.asarray(sine, dtype=float)
    with np.errstate(invalid="ignore"):
        cos_f = np.sqrt(1.0 - sine * sine)
        smallest = k * np.abs(cos_f - math.cos(math.radians(incidence_deg)))
        largest = k * (cos_f + math.cos(math.radians(incidence_deg)))
        q_parallel = np.abs(np.asarray(q_parallel, dtype=float))
        return np.isfinite(sine) & (q_parallel >= smallest - 1e-12) & (q_parallel <= largest)


__all__ = ["CHANNELS", "REFLECTED", "TRANSMITTED", "exit_sine", "incident_kz", "map_qz", "observed_qz", "reachable"]
