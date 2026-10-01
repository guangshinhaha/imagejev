import json

import pytest

from imagejev.bench.gate import evaluate_gate, main


def rows(per_domain):
    out = []
    for domain, (n, acc, ll, ece) in per_domain.items():
        out.append(
            {"domain": domain, "qtype": "all", "n": n, "accuracy": acc, "log_loss": ll, "ece": ece}
        )
    return out


def results(model, **splits):
    return {"model": model, "splits": {s: {"rows": rows(d)} for s, d in splits.items()}}


BASE = {
    "photo": (100, 0.60, 0.80, 0.05),
    "document": (100, 0.62, 0.78, 0.06),
    "screenshot": (100, 0.67, 0.70, 0.04),
}


def better(delta_ll=-0.1, delta_ece=-0.01, delta_acc=0.05):
    return {
        d: (n, a + delta_acc, ll + delta_ll, e + delta_ece) for d, (n, a, ll, e) in BASE.items()
    }


def test_passes_when_better_in_every_domain():
    r = evaluate_gate(results("c", val=better()), results("b", val=BASE), "val")
    assert r.passed and r.criterion1_ok and r.criterion2_ok
    assert [c.domain for c in r.criterion1] == ["photo", "document", "screenshot"]
    md = r.to_markdown("cand", "base")
    assert "**PASS**" in md and "cand vs base" in md


def test_one_domain_worse_on_log_loss_fails_criterion_1():
    cand = better()
    cand["document"] = (100, 0.70, 0.90, 0.01)  # higher log loss than the baseline's 0.78
    r = evaluate_gate(results("c", val=cand), results("b", val=BASE), "val")
    assert not r.criterion1_ok and not r.passed
    bad = next(c for c in r.criterion1 if c.domain == "document")
    assert not bad.log_loss_ok and bad.ece_ok
    assert "FAIL" in r.to_markdown()


def test_worse_ece_alone_fails():
    cand = better(delta_ece=+0.02)
    r = evaluate_gate(results("c", val=cand), results("b", val=BASE), "val")
    assert not r.criterion1_ok and all(c.log_loss_ok and not c.ece_ok for c in r.criterion1)


def test_criterion_2_checks_every_split_both_runs_have():
    cand = {"val": better(), "test-images": better(delta_acc=0.0), "test-tasks": better()}
    cand["test-images"]["screenshot"] = (100, 0.60, 0.5, 0.01)  # below the baseline's 0.67 accuracy
    base = {"val": BASE, "test-images": BASE, "test-tasks": BASE}
    r = evaluate_gate(results("c", **cand), results("b", **base), "val")
    assert r.criterion1_ok and not r.criterion2_ok and not r.passed
    failing = [(c["split"], c["domain"]) for c in r.criterion2 if not c["ok"]]
    assert failing == [("test-images", "screenshot")]
    assert len(r.criterion2) == 9  # 3 splits x 3 domains


def test_equal_accuracy_passes_criterion_2_and_a_split_only_one_run_has_is_skipped():
    cand = {"val": better(delta_acc=0.0), "test-styles": better()}
    r = evaluate_gate(results("c", **cand), results("b", val=BASE), "val")
    assert r.criterion2_ok and {c["split"] for c in r.criterion2} == {"val"}


def test_thin_or_missing_domains_are_reported_not_judged():
    cand = better()
    cand["screenshot"] = (5, 0.99, 0.01, 0.0)  # far too few questions to trust
    r = evaluate_gate(results("c", val=cand), results("b", val=BASE), "val")
    assert not r.criterion1_ok and any("screenshot" in m for m in r.missing)
    assert "Missing" in r.to_markdown()
    nothing = evaluate_gate(results("c", val=better()), results("b", val=BASE), "test-tasks")
    assert not nothing.passed and len(nothing.missing) == 3


def test_cli_exit_codes_and_report(tmp_path, monkeypatch, capsys):
    (tmp_path / "c.json").write_text(json.dumps(results("cand", val=better())))
    (tmp_path / "b.json").write_text(json.dumps(results("base", val=BASE)))
    out = tmp_path / "gate.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gate",
            "--candidate",
            str(tmp_path / "c.json"),
            "--baseline",
            str(tmp_path / "b.json"),
            "--out",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0 and "PASS" in out.read_text()
    (tmp_path / "c.json").write_text(
        json.dumps(results("cand", val={**BASE, "photo": (100, 0.5, 0.95, 0.2)}))
    )
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
