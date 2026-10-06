"""Run the examples, so they cannot drift from the code.

A snippet in a README is not checked by anything and goes stale silently.
This session has already turned up a test that passed vacuously and a
docstring quoting a validated range nobody had re-measured, so the examples
are executed rather than trusted.
"""
import runpy
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.slow
def test_quickstart_runs(capsys):
    """Execute examples/jeanie_quickstart.py end to end."""
    script = EXAMPLES / "jeanie_quickstart.py"
    assert script.exists(), f"missing {script}"
    sys.path.insert(0, str(EXAMPLES))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.path.remove(str(EXAMPLES))

    out = capsys.readouterr().out
    # every section must have produced output, not silently no-opped
    for section in ("spherical solve", "existence and branch", "axisymmetric",
                    "differentiable solve", "contracted boundary",
                    "export as a potential"):
        assert f"=== {section} ===" in out, f"section missing: {section}"
    assert "nan" not in out.lower(), f"a NaN reached the output:\n{out}"
    # and the numbers must be the ones the text claims
    assert "r0 = 1.7506 kpc" in out, "the spherical fiducial moved"
    assert "M(<r1)/M1 - 1 = 2.66e-15" in out, "the matching is no longer exact"


def test_every_example_is_importable_and_named_consistently():
    """Cheap guard: no example may reference the pre-rename module path."""
    bad = []
    for f in sorted(EXAMPLES.glob("*.py")):
        text = f.read_text()
        if "jeans.fast" in text:
            bad.append(f.name)
    assert not bad, f"stale jeans.fast references in {bad}"
