import json

import pytest

from imagejev.bench.compare import cell, compare_table, latency_line, main


def res(acc, ll, ece, n=100, lat=None):
    rows = [
        {"domain": d, "qtype": "all", "n": n, "accuracy": acc, "log_loss": ll, "ece": ece}
        for d in ("photo", "document", "all")
    ]
    return {"splits": {"val": {"rows": rows}}, "latency": lat}


def test_cell_and_missing_cell():
    assert cell({"accuracy": 0.6, "log_loss": 0.7, "ece": 0.05}) == "0.600 / 0.700 / 0.050"
    assert cell(None) == "-"


def test_table_has_a_row_per_present_domain_and_a_column_per_model():
    t = compare_table({"base": res(0.5, 0.8, 0.1), "ours": res(0.6, 0.7, 0.05)}, "val")
    lines = t.splitlines()
    assert lines[0] == "**val**" and lines[2] == "| domain | n | base | ours |"
    assert "| photo | 100 | 0.500 / 0.800 / 0.100 | 0.600 / 0.700 / 0.050 |" in t
    assert "screenshot" not in t  # no such domain in the data: no row
    assert t.endswith("_cells are accuracy / log loss / ECE_")


def test_unknown_split_and_missing_models_rows():
    with pytest.raises(KeyError):
        compare_table({"a": res(0.5, 0.8, 0.1)}, "nope")
    a = res(0.5, 0.8, 0.1)
    b = {
        "splits": {
            "val": {
                "rows": [
                    r for r in res(0.6, 0.7, 0.05)["splits"]["val"]["rows"] if r["domain"] == "all"
                ]
            }
        }
    }
    t = compare_table({"a": a, "b": b}, "val")
    assert "| photo | 100 | 0.500 / 0.800 / 0.100 | - |" in t  # a model without that row shows '-'


def test_latency_line():
    lat = {"cold": {"p50_ms": 61.2}, "warm": {"p50_ms": 0.7}}
    assert latency_line({"a": res(0.5, 0.8, 0.1, lat=lat), "b": res(0.5, 0.8, 0.1)}).endswith(
        "a: cold 61 ms, warm 0.7 ms; b: n/a"
    )
    assert "warm" not in latency_line(
        {"x": res(0.5, 0.8, 0.1, lat={"cold": {"p50_ms": 5.0}, "warm": None})}
    )


def test_cli_prints_every_split(tmp_path, monkeypatch, capsys):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(res(0.5, 0.8, 0.1)))
    monkeypatch.setattr("sys.argv", ["compare", "--results", f"base={p}"])
    main()
    out = capsys.readouterr().out
    assert "**val**" in out and "base" in out and "Latency p50" in out
