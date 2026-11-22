"""Interpret an explicitly identified GISAXS substrate; require complete optics for each mentioned film."""

import math
from collections.abc import Mapping

MATERIALS = {"si": "Si", "silicon": "Si", "sio2": "SiO2", "silica": "SiO2", "silicon dioxide": "SiO2",
             "fused silica": "SiO2", "glass": "glass", "plate glass": "glass", "glass (plate)": "glass",
             "au": "Au", "gold": "Au"}
UNKNOWN_MATERIAL = "Unsupported substrate material {material}."
BAD_BASE = "The substrate needs a known material or finite optical constants δ > 0 and β ≥ 0."
BAD_FILM = "Each specified film needs its thickness in nm and finite optical constants δ and β."
BAD_DESCRIPTION = "The substrate description contains unrecognized layer or optical fields."


def material_name(value):
    key = " ".join(str(value).casefold().replace("₂", "2").split())
    if key not in MATERIALS:
        raise ValueError(UNKNOWN_MATERIAL.format(material=value))
    return MATERIALS[key]


def _constants(record, *, film=False):
    keys = ("delta", "beta", "thickness_nm") if film else ("delta", "beta")
    try:
        values = {key: float(record[key]) for key in keys}
        valid = all(not isinstance(record[k], bool) and math.isfinite(v) and v >= 0 for k, v in values.items())
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    if not valid or (values["thickness_nm"] if film else values["delta"]) <= 0:
        raise ValueError(BAD_FILM if film else BAD_BASE)
    return values


def substrate_record(raw):
    if isinstance(raw, str):
        return {"material": material_name(raw)}
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(BAD_BASE)
    if raw.keys() - {"material", "delta", "beta", "layers", "top_layer"} or ("layers" in raw and "top_layer" in raw):
        raise ValueError(BAD_DESCRIPTION)
    if "delta" in raw or "beta" in raw or raw.get("material", "custom") == "custom":
        support = {"material": "custom", **_constants(raw)}
    else:
        support = {"material": material_name(raw["material"])}
    layers = [raw["top_layer"]] if "top_layer" in raw else raw.get("layers", ())
    if not isinstance(layers, (tuple, list)):
        raise ValueError(BAD_FILM)
    if layers:
        support["layers"] = []
        for layer in layers:
            if not isinstance(layer, Mapping) or layer.keys() - {"delta", "beta", "thickness_nm"}:
                raise ValueError(BAD_FILM)
            support["layers"].append(_constants(layer, film=True))
    return support
