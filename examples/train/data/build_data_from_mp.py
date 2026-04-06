from __future__ import annotations

import os

from monty.serialization import dumpfn
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
from tqdm.auto import tqdm

API_KEY = os.getenv("MP_API_KEY")
PROPERTY = "formation_energy_per_atom"


if not API_KEY:
    raise RuntimeError(
        "Missing MP_API_KEY environment variable. "
        "Please export MP_API_KEY before running build_data.py."
    )

mpr = MPRester(API_KEY, mute_progress_bars=False, use_document_model=False)

available_fields = mpr.materials.summary.available_fields
if PROPERTY not in available_fields:
    raise ValueError(
        f"Invalid PROPERTY: {PROPERTY!r}. "
        f"Choose one of summary.available_fields, e.g. {sorted(available_fields)}"
    )
search_fields = ["material_id", "structure", PROPERTY]

docs = mpr.materials.summary.search(
    deprecated=False,
    all_fields=False,
    fields=search_fields,
)

sample_ids = []
structures = []
targets = []
for doc in tqdm(docs, desc="Building dataset", dynamic_ncols=True):
    material_id = doc["material_id"]
    structure = Structure.from_dict(doc["structure"])
    target = doc[PROPERTY]
    if target is not None:
        sample_ids.append(material_id)
        structures.append(structure)
        targets.append(target)

data = {
    "sample_ids": sample_ids,
    "structures": structures,
    "targets": targets,
}
dumpfn(obj=data, fn=f"MP_{PROPERTY}.json")
