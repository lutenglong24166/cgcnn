from __future__ import annotations

import pandas as pd
from monty.serialization import dumpfn
from mp_api.client import MPRester

API_KEY = "h7IsuVkIMpekrsJgfREs5DFd5mOnSoBT"
DATA_FRAC = 0.001
SEED = 42


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


mpr = MPRester(API_KEY, mute_progress_bars=False)
df = pd.read_csv("2025-02-01-mp-energies.csv")
sample_df = df.sample(frac=DATA_FRAC, random_state=SEED)

mp_ids = sample_df["material_id"].dropna().astype(str).unique().tolist()
print(f"number of unique material IDs: {len(mp_ids)}")

chunk_size = 9000  # mp_api limit is 10000 IDs per request
docs = []
for ids_chunk in chunked(mp_ids, chunk_size):
    docs_chunk = mpr.materials.search(
        material_ids=ids_chunk,
        fields=["material_id", "structure"],
    )
    docs.extend(docs_chunk)

structure_map = {
    str(doc.material_id): doc.structure for doc in docs if doc.structure is not None
}

valid_df = sample_df[sample_df["material_id"].isin(structure_map)].reset_index(
    drop=True
)

sample_ids = valid_df["material_id"].tolist()
structures = [structure_map[mid] for mid in sample_ids]
targets = valid_df["formation_energy_per_atom"].astype(float).tolist()

data = {
    "sample_ids": sample_ids,
    "structures": structures,
    "targets": targets,
}
dumpfn(obj=data, fn="formation_energies.json")
