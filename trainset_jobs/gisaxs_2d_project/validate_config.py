from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from trainset.config import load_project_config, validate_project_config
from trainset.geometry import roi_to_spherical_ranges
cfg = load_project_config(ROOT / "config.yaml")
valid, errors, warnings = validate_project_config(cfg)
print("valid=", valid)
print("spherical_ranges=", roi_to_spherical_ranges(cfg))
for item in warnings: print("WARNING:", item)
for item in errors: print("ERROR:", item)
raise SystemExit(0 if valid else 1)
