from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.contract import (
    CYLINDER,
    SPHERE,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    NUM_TOPOLOGIES,
    full_component_bounds,
    gui_component_to_latent,
    latent_component_to_gui,
    topology_from_id,
)
from PosteriorV8.profiled_forward import build_design_matrix
from PosteriorV8.profiled_refinement import ResolutionBounds, refine_profiled_branch
from PosteriorV8.branch_codec import (
    INACTIVE_UNIT_VALUE,
    UNIT_CUBE_DIMENSIONS,
    ProfiledBranchCodec,
)
from PosteriorV8.proposal_sampling import (
    generate_profiled_branch_seed_at_index,
    generate_profiled_branch_seeds,
)


def _bounds(shape: str, *, optional_d: bool = False) -> GuiComponentBounds:
    common = {
        "shape": shape,
        "R": ClosedInterval(4.0, 18.0),
        "sigma_R": (
            ClosedInterval(0.05, 0.35) if shape == VERTICAL_CYLINDER else ClosedInterval(0.5, 3.0)
        ),
    }
    if shape == CYLINDER:
        common.update(
            h=ClosedInterval(7.0, 24.0),
            sigma_h=ClosedInterval(0.7, 4.0),
        )
    if optional_d:
        common.update(
            D=ClosedInterval(18.0, 90.0),
            sigma_D=ClosedInterval(1.8, 12.0),
            allow_D_absent=True,
        )
    return GuiComponentBounds(**common)


class PosteriorV8ProposalSamplingTests(unittest.TestCase):
    def test_indexed_materialization_matches_the_existing_batch_stream_exactly(self):
        topology = (SPHERE, CYLINDER)
        bounds = tuple(_bounds(shape, optional_d=True) for shape in topology)
        codec = ProfiledBranchCodec.build(topology, bounds, (True, False))
        batch = generate_profiled_branch_seeds(
            topology,
            bounds,
            (True, False),
            seed=314159,
            count=9,
        )

        indexed = tuple(
            generate_profiled_branch_seed_at_index(
                codec,
                seed=314159,
                sequence_index=index,
            )
            for index in range(9)
        )

        self.assertEqual(indexed, batch)

    def test_repeated_calls_are_identical_and_obey_all_coupled_ranges(self):
        topology = (SPHERE, CYLINDER, VERTICAL_CYLINDER)
        bounds = tuple(_bounds(shape, optional_d=True) for shape in topology)
        resolution_bounds = ResolutionBounds(
            ClosedInterval(0.002, 0.08),
            ClosedInterval(1.5, 9.0),
        )
        kwargs = {
            "topology": topology,
            "component_bounds": bounds,
            "d_present": (True, True, False),
            "resolution_bounds": resolution_bounds,
            "seed": 20260902,
            "count": 16,
        }

        first = generate_profiled_branch_seeds(**kwargs)
        second = generate_profiled_branch_seeds(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        codec = ProfiledBranchCodec.build(
            topology,
            bounds,
            (True, True, False),
            resolution_bounds=resolution_bounds,
        )
        for branch_seed in first:
            self.assertEqual(branch_seed.topology, topology)
            self.assertEqual(branch_seed.d_present, (True, True, False))
            self.assertIsNotNone(branch_seed.resolution_seed)
            self.assertTrue(resolution_bounds.contains(branch_seed.resolution_seed))
            for gui_bounds, latent, present in zip(
                bounds, branch_seed.seed_components, branch_seed.d_present
            ):
                gui = latent_component_to_gui(latent)
                self.assertEqual(gui.D is not None, present)
                self.assertTrue(gui_bounds.R.low <= gui.R <= gui_bounds.R.high)
                self.assertTrue(gui_bounds.sigma_R.low <= gui.sigma_R <= gui_bounds.sigma_R.high)
                if gui.h is not None:
                    self.assertTrue(gui_bounds.h.low <= gui.h <= gui_bounds.h.high)
                    self.assertTrue(
                        gui_bounds.sigma_h.low <= gui.sigma_h <= gui_bounds.sigma_h.high
                    )
                if gui.D is not None:
                    self.assertTrue(gui_bounds.D.low <= gui.D <= gui_bounds.D.high)
                    self.assertTrue(
                        gui_bounds.sigma_D.low <= gui.sigma_D <= gui_bounds.sigma_D.high
                    )
                    exclusion = (
                        np.hypot(2.0 * gui.R, gui.h) if gui.shape == CYLINDER else 2.0 * gui.R
                    )
                    self.assertGreater(gui.D, 1.001 * exclusion)

            self.assertEqual(len(branch_seed.unit_cube), UNIT_CUBE_DIMENSIONS)
            self.assertEqual(len(branch_seed.active_mask), UNIT_CUBE_DIMENSIONS)
            self.assertEqual(branch_seed.active_mask, codec.active_mask)
            self.assertEqual(sum(branch_seed.active_mask), 14)
            self.assertTrue(
                all(
                    value == INACTIVE_UNIT_VALUE
                    for value, active in zip(branch_seed.unit_cube, branch_seed.active_mask)
                    if not active
                )
            )

            encoded = codec.encode(branch_seed.seed_components, branch_seed.resolution_seed)
            public_encoded = ProfiledBranchCodec.build(
                topology,
                bounds,
                (True, True, False),
                resolution_bounds=resolution_bounds,
            ).encode(branch_seed.seed_components, branch_seed.resolution_seed)
            self.assertEqual(encoded, public_encoded)
            np.testing.assert_allclose(
                np.asarray(encoded.unit_cube)[np.asarray(encoded.active_mask)],
                np.asarray(branch_seed.unit_cube)[np.asarray(branch_seed.active_mask)],
                rtol=0.0,
                atol=5e-15,
            )
            decoded_components, decoded_resolution = codec.decode(encoded)
            for expected, decoded in zip(branch_seed.seed_components, decoded_components):
                np.testing.assert_allclose(
                    [
                        value
                        for value in expected.__dict__.values()
                        if isinstance(value, (int, float))
                    ],
                    [
                        value
                        for value in decoded.__dict__.values()
                        if isinstance(value, (int, float))
                    ],
                    rtol=0.0,
                    atol=2e-14,
                )
            self.assertAlmostEqual(
                decoded_resolution.sigma_res,
                branch_seed.resolution_seed.sigma_res,
                places=14,
            )
            self.assertAlmostEqual(
                decoded_resolution.nu_res,
                branch_seed.resolution_seed.nu_res,
                places=14,
            )

    def test_every_catalog_topology_keeps_canonical_id_and_order(self):
        for topology_id in range(NUM_TOPOLOGIES):
            with self.subTest(topology_id=topology_id):
                topology = topology_from_id(topology_id)
                bounds = tuple(
                    full_component_bounds(shape, d_policy="absent") for shape in topology
                )
                result = generate_profiled_branch_seeds(
                    topology,
                    bounds,
                    tuple(False for _ in topology),
                    seed=1000 + topology_id,
                    count=1,
                )
                self.assertEqual(result[0].topology_id, topology_id)
                self.assertEqual(tuple(item.shape for item in result[0].seed_components), topology)

                required_bounds = tuple(
                    full_component_bounds(shape, d_policy="required") for shape in topology
                )
                required = generate_profiled_branch_seeds(
                    topology,
                    required_bounds,
                    tuple(True for _ in topology),
                    seed=2000 + topology_id,
                    count=2,
                )
                codec = ProfiledBranchCodec.build(
                    topology,
                    required_bounds,
                    tuple(True for _ in topology),
                )
                for branch_seed in required:
                    recovered = codec.encode(branch_seed.seed_components)
                    active = np.asarray(branch_seed.active_mask)
                    np.testing.assert_allclose(
                        np.asarray(recovered.unit_cube)[active],
                        np.asarray(branch_seed.unit_cube)[active],
                        rtol=0.0,
                        atol=5e-12,
                    )
                    for component in branch_seed.seed_components:
                        gui = latent_component_to_gui(component)
                        exclusion = (
                            np.hypot(2.0 * gui.R, gui.h) if gui.shape == CYLINDER else 2.0 * gui.R
                        )
                        self.assertGreater(gui.D, 1.001 * exclusion)

    def test_infeasible_or_contradictory_d_branch_fails_without_clipping(self):
        impossible_hard_core = GuiComponentBounds(
            SPHERE,
            R=ClosedInterval(20.0, 30.0),
            sigma_R=ClosedInterval(2.0, 3.0),
            D=ClosedInterval(3.0, 30.0),
            sigma_D=ClosedInterval(0.3, 3.0),
            allow_D_absent=True,
        )
        with self.assertRaisesRegex(ValueError, "no geometry satisfying hard-core spacing"):
            generate_profiled_branch_seeds(
                (SPHERE,),
                (impossible_hard_core,),
                True,
                seed=1,
                count=4,
            )

        absent_only = _bounds(SPHERE)
        with self.assertRaisesRegex(ValueError, "has no D user ranges"):
            generate_profiled_branch_seeds((SPHERE,), (absent_only,), True, seed=1, count=1)

        required = GuiComponentBounds(
            SPHERE,
            R=ClosedInterval(4.0, 8.0),
            sigma_R=ClosedInterval(0.5, 1.5),
            D=ClosedInterval(20.0, 40.0),
            sigma_D=ClosedInterval(2.0, 5.0),
        )
        with self.assertRaisesRegex(ValueError, "conflicts with required D ranges"):
            generate_profiled_branch_seeds((SPHERE,), (required,), False, seed=1, count=1)

        hard_core_violating_seed = gui_component_to_latent(
            GuiComponentParameters(
                SPHERE,
                R=10.0,
                sigma_R=1.0,
                D=20.0,
                sigma_D=2.0,
            )
        )
        broad_required = GuiComponentBounds(
            SPHERE,
            R=ClosedInterval(8.0, 12.0),
            sigma_R=ClosedInterval(0.8, 1.2),
            D=ClosedInterval(18.0, 40.0),
            sigma_D=ClosedInterval(1.8, 4.0),
        )
        codec = ProfiledBranchCodec.build((SPHERE,), (broad_required,), True)
        with self.assertRaisesRegex(ValueError, "violates hard-core spacing"):
            codec.encode((hard_core_violating_seed,))

    def test_noncanonical_or_misaligned_topology_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical catalog"):
            generate_profiled_branch_seeds(
                (CYLINDER, SPHERE),
                (_bounds(CYLINDER), _bounds(SPHERE)),
                (False, False),
                seed=1,
                count=1,
            )
        with self.assertRaisesRegex(ValueError, "same canonical topology order"):
            generate_profiled_branch_seeds(
                (SPHERE, CYLINDER),
                (_bounds(CYLINDER), _bounds(SPHERE)),
                (False, False),
                seed=1,
                count=1,
            )

    def test_fixed_user_ranges_use_the_unique_canonical_inverse(self):
        bounds = GuiComponentBounds(
            SPHERE,
            R=ClosedInterval(10.0, 10.0),
            sigma_R=ClosedInterval(1.0, 1.0),
        )
        resolution_bounds = ResolutionBounds(
            ClosedInterval(0.02, 0.02),
            ClosedInterval(4.0, 4.0),
        )
        branch_seed = generate_profiled_branch_seeds(
            (SPHERE,),
            (bounds,),
            False,
            resolution_bounds=resolution_bounds,
            seed=91,
            count=1,
        )[0]
        codec = ProfiledBranchCodec.build(
            (SPHERE,),
            (bounds,),
            False,
            resolution_bounds=resolution_bounds,
        )

        self.assertEqual(
            branch_seed.coordinates,
            codec.encode(branch_seed.seed_components, branch_seed.resolution_seed),
        )
        for index in (0, 1, 24, 25):
            self.assertTrue(branch_seed.active_mask[index])
            self.assertEqual(branch_seed.unit_cube[index], INACTIVE_UNIT_VALUE)

    def test_seed_payload_can_be_passed_directly_to_refinement(self):
        q = np.geomspace(0.01, 1.0, 80)
        truth = GuiComponentParameters(SPHERE, R=10.0, sigma_R=1.0)
        intensity = build_design_matrix(q, (truth,)) @ np.asarray([0.01, 2.0])
        bounds = GuiComponentBounds(
            SPHERE,
            R=ClosedInterval(8.0, 14.0),
            sigma_R=ClosedInterval(0.5, 2.0),
        )
        branch_seed = generate_profiled_branch_seeds((SPHERE,), (bounds,), False, seed=22, count=1)[
            0
        ]

        result = refine_profiled_branch(
            q,
            intensity,
            **branch_seed.refinement_kwargs(),
            max_nfev=2,
        )

        self.assertEqual(result.initial_latent_components, branch_seed.seed_components)
        self.assertTrue(result.bounds_satisfied)

    def test_invalid_seed_count_and_branch_flags_are_rejected(self):
        bounds = (_bounds(SPHERE), _bounds(CYLINDER))
        topology = (SPHERE, CYLINDER)
        for kwargs in (
            {"seed": -1, "count": 1, "d_present": (False, False)},
            {"seed": 1, "count": 0, "d_present": (False, False)},
            {"seed": 1, "count": 1, "d_present": False},
            {"seed": 1, "count": 1, "d_present": (False, 0)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                generate_profiled_branch_seeds(
                    topology,
                    bounds,
                    seed=kwargs["seed"],
                    count=kwargs["count"],
                    d_present=kwargs["d_present"],
                )

        with self.assertRaisesRegex(ValueError, "versioned V8 resolution domain"):
            generate_profiled_branch_seeds(
                topology,
                bounds,
                (False, False),
                resolution_bounds=ResolutionBounds(
                    ClosedInterval(0.002, 0.4),
                    ClosedInterval(1.0, 10.0),
                ),
                seed=1,
                count=1,
            )


if __name__ == "__main__":
    unittest.main()
