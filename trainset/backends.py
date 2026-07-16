from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class JobStatus:
    job_id: str
    state: str
    elapsed: str = ""
    max_rss: str = ""
    exit_code: str = ""
    raw: str = ""


class LocalBackend:
    def run_generate(self, package_dir: str | Path, samples: int) -> subprocess.Popen[str]:
        root = Path(package_dir)
        return subprocess.Popen(
            [str(Path(shutil.which("python") or "python")), "generate_dataset.py", "--samples", str(samples), "--mode", "full"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def run_training(self, package_dir: str | Path) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [str(Path(shutil.which("python") or "python")), "train.py"],
            cwd=Path(package_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


class SlurmBackend:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        hpc = config["hpc"]
        self.target = f"{hpc['user']}@{hpc['host']}"
        self.remote_path = str(hpc["remote_path"]).rstrip("/")

    def _run(self, command: list[str], timeout: int = 60) -> str:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if completed.returncode:
            raise RuntimeError((completed.stderr or completed.stdout).strip())
        return completed.stdout.strip()

    def connection_check(self) -> str:
        remote = shlex.quote(self.remote_path)
        command = f"mkdir -p {remote} && test -w {remote} && printf GIMAP_OK"
        return self._run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", self.target, command])

    def upload(self, package_dir: str | Path) -> str:
        if shutil.which("rsync"):
            return self._run(["rsync", "-av", f"{Path(package_dir)}/", f"{self.target}:{self.remote_path}/"], timeout=3600)
        return self._run(["scp", "-r", str(Path(package_dir)), f"{self.target}:{str(Path(self.remote_path).parent)}/"], timeout=3600)

    def submit(self) -> Dict[str, str]:
        remote = shlex.quote(self.remote_path)
        command = (
            f"cd {remote} && mkdir -p logs results dataset && "
            "GEN_JOB=$(sbatch --parsable slurm_generate.sh) && "
            "TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$GEN_JOB slurm_train.sh) && "
            "printf '%s %s' \"$GEN_JOB\" \"$TRAIN_JOB\""
        )
        output = self._run(["ssh", self.target, command])
        values = output.split()
        if len(values) < 2:
            raise RuntimeError(f"Unable to parse Slurm job IDs: {output}")
        return {"generate_job_id": values[0], "train_job_id": values[1]}

    def query(self, job_id: str) -> JobStatus:
        if not re.fullmatch(r"[0-9]+", str(job_id)):
            raise ValueError(f"Invalid Slurm job ID: {job_id}")
        fmt = "JobIDRaw,State,Elapsed,MaxRSS,ExitCode"
        output = self._run(["ssh", self.target, f"sacct -n -P -j {job_id} --format={fmt}"])
        line = next((item for item in output.splitlines() if item.strip()), "")
        parts = line.split("|")
        return JobStatus(job_id, parts[1] if len(parts) > 1 else "UNKNOWN", *(parts[2:5] + [""] * 3)[:3], raw=output)

    def tail(self, job_id: str, lines: int = 100) -> str:
        if not re.fullmatch(r"[0-9]+", str(job_id)):
            raise ValueError(f"Invalid Slurm job ID: {job_id}")
        remote = shlex.quote(self.remote_path)
        return self._run(["ssh", self.target, f"cd {remote} && tail -n {int(lines)} logs/*{job_id}*.out 2>/dev/null || true"])

    def download_results(self, local_dir: str | Path) -> str:
        destination = Path(local_dir)
        destination.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            return self._run(["rsync", "-av", f"{self.target}:{self.remote_path}/results/", f"{destination}/"], timeout=3600)
        return self._run(["scp", "-r", f"{self.target}:{self.remote_path}/results/.", str(destination)], timeout=3600)


def read_metrics(path: str | Path) -> list[Dict[str, Any]]:
    records = []
    source = Path(path)
    if not source.exists():
        return records
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records
