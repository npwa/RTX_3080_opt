import json
from pathlib import Path

import pytest

from rtxopt.runners import env_manifest as em

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_gpu_query():
    fields = em.GPU_FIELDS
    line = ", ".join(str(i) for i in range(len(fields)))
    gpus = em.parse_gpu_query(line + "\n")
    assert gpus[0]["name"] == "0" and gpus[0]["memory.total"] == str(len(fields) - 1)


def test_parse_gpu_query_rejects_bad_columns():
    with pytest.raises(ValueError):
        em.parse_gpu_query("a, b")


def test_parse_nvcc_version():
    text = "Cuda compilation tools, release 12.0, V12.0.140\nBuild x"
    assert em.parse_nvcc_version(text) == "12.0"
    assert em.parse_nvcc_version("nothing") is None


def test_collect_is_json_serialisable_without_gpu(monkeypatch):
    monkeypatch.setattr(em, "_run", lambda *a, **k: None)
    manifest = em.collect()
    assert manifest["gpus"] == [] and manifest["cuda_toolkit"] is None
    json.dumps(manifest)


def test_cli_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "_run", lambda *a, **k: None)
    out = tmp_path / "sub" / "env.json"
    assert em.main(["--project", str(tmp_path), "-o", str(out)]) == 0
    assert "timestamp_utc" in json.loads(out.read_text())


def test_fixtures_present():
    assert (FIXTURES / "ncu" / "graphopt_on_ncu.csv").stat().st_size > 0
    assert json.loads((FIXTURES / "bench" / "baseline_raw.json").read_text())
