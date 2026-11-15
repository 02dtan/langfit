"""The golden reference for the browser energy-landscape engine cannot rot, and
input drift is reported explicitly.

Two concerns are kept separate:

* ``test_golden_derivation_is_deterministic`` — the golden is self-contained:
  every case embeds the exact inputs (couplings, basis) its outputs were derived
  from. This test recomputes each case *from those embedded inputs* (via
  ``engine_golden._case_from_V``) and requires it to equal the stored case
  byte-for-byte. It never reads ``data/derived/landscape_artifact.npz``, so it
  tests only the output derivation and passes regardless of cache drift. This is
  the "golden cannot rot" guarantee.

* ``test_golden_inputs_match_cache`` — cheap: the stored ``meta.input_hash`` (a
  SHA-256 of the fitted couplings) must equal the hash of the current cache. If
  the couplings were refit, this fails with an explicit message telling the
  maintainer to regenerate ``scripts/engine_golden.py``, rather than surfacing as
  a baffling per-field diff.

The JS side is exercised by ``tests/js/engine_parity.mjs`` (run under Bun), not
here. The derivation recomputation is intentionally a full pass (~10 s).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import engine_golden as eg  # noqa: E402


def test_golden_derivation_is_deterministic():
    stored = json.loads(eg.OUT.read_text())
    cases = stored["cases"]
    assert cases, "stored golden has no cases"

    for c in cases:
        V = eg.b64_decode(c["V"], np.float64)
        if c.get("coords"):
            regen = eg._case_from_V(c["model"], V, c["basis"], c["G"],
                                    coords=eg.b64_decode(c["coords"], np.float32))
        else:
            regen = eg._case_from_V(c["model"], V, c["basis"], c["G"])
        assert regen == c, f"case {c['model']}/{c['basis']} G{c['G']} differs"


def test_golden_inputs_match_cache():
    stored = json.loads(eg.OUT.read_text())
    record = stored["meta"]["input_hash"]
    current = eg.input_hash(np.load(eg.NPZ))
    assert current == record, (
        "golden inputs are stale: the fitted couplings in "
        f"{eg.NPZ.name} changed (hash {current[:12]}... != recorded {record[:12]}...). "
        "Regenerate the golden with `python scripts/engine_golden.py`."
    )
