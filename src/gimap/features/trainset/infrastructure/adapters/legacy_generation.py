"""Existing DatasetGenerator 的 infrastructure adapter。"""

from __future__ import annotations

from ...application import GeneratedTrainset


class LegacyDatasetGenerationAdapter:
    def __init__(self, simulation_port):
        self.simulation_port = simulation_port

    def generate(self, request, *, on_progress=None, pause=None):
        from trainset.generator import DatasetGenerator

        generator = DatasetGenerator(
            request.config,
            simulation_port=self.simulation_port,
        )
        if request.output_dir is not None:
            files = generator.write_hdf5_shards(
                request.output_dir,
                request.sample_count,
                mode=request.mode,
                progress=on_progress,
                pause=pause,
            )
            return GeneratedTrainset(files=tuple(files))
        value = generator.generate(
            request.sample_count,
            mode=request.mode,
            progress=on_progress,
            pause=pause,
        )
        return GeneratedTrainset(value=value)
