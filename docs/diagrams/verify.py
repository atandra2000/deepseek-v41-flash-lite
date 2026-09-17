#!/usr/bin/env python3
"""Check the four delivered diagrams with Archify, without rerendering HTML."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ARCHIFY = Path(os.environ.get("ARCHIFY", Path.home() / ".agents/skills/archify/bin/archify.mjs"))
DIAGRAMS = (
    ("architecture-model", "architecture", "model.architecture.json"),
    ("architecture-csa2", "architecture", "architecture-csa2.json"),
    ("workflow-training-loop", "workflow", "workflow-training-loop.json"),
    ("dataflow-data-pipeline", "dataflow", "dataflow-data-pipeline.json"),
)


def digest(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def main():
    results = []
    for name, kind, specification in DIAGRAMS:
        try:
            delivery = json.loads((ROOT / f"{name}.delivery.json").read_text())
            assert delivery["ok"] and delivery["validation"] == {
                "checksPassed": 9, "checkCount": 9, "compositionProfile": "showcase",
                "compositionStatus": "pass", "errors": 0, "warnings": 0,
            }, "Delivery must pass all nine showcase checks"
            assert digest(ROOT / specification) == delivery["specification"], "Specification changed since delivery"
            assert digest(ROOT / f"{name}.html") == delivery["artifact"], "HTML changed since delivery"
            validation = subprocess.run(
                ["node", str(ARCHIFY), "validate", kind, str(ROOT / specification), "--quality", "showcase", "--json"],
                capture_output=True, text=True, timeout=60,
            )
            if validation.returncode:
                raise ValueError(validation.stdout + validation.stderr)
            browser = subprocess.run(
                ["node", str(ARCHIFY), "visual-check", str(ROOT / f"{name}.html"), "--json"],
                capture_output=True, text=True, timeout=180,
            )
            evidence = json.loads(browser.stdout)
            passed = (browser.returncode == 0 and evidence.get("ok") is True
                      and evidence.get("status") == "pass"
                      and evidence.get("artifact", {}).get("sha256") == delivery["artifact"]["sha256"])
            results.append({"diagram": name, "passed": passed, "exit_code": browser.returncode,
                            "specification": delivery["specification"], "artifact": delivery["artifact"],
                            "browser_receipt": f"{name}.visual-check.json",
                            "diagnostics": evidence.get("diagnostics", []), "stderr": browser.stderr})
        except (OSError, ValueError, KeyError, AssertionError, subprocess.TimeoutExpired) as error:
            results.append({"diagram": name, "passed": False, "error": str(error)})
    report = {"passed": all(r["passed"] for r in results), "diagrams": results,
              "visual_review": "not assessed by this automated command"}
    (ROOT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
