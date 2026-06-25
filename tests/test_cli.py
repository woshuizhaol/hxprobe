import hxprobe
from hxprobe import cli


def test_cli_example_runs(capsys):
    rc = cli.main(["example"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Spearman" in out
    assert "ubiquitin" in out.lower()


def test_cli_score_to_csv(tmp_path):
    out = tmp_path / "dg.csv"
    rc = cli.main(["score", hxprobe.example_ensemble_path(),
                   "--protonate", "none", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    text = out.read_text()
    assert "dGopen_kcal" in text.splitlines()[0]
    assert len(text.splitlines()) > 30
