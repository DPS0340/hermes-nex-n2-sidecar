#!/usr/bin/env python3
from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
LABEL = "ai.hermes.nex-n2-sidecar"
PLIST = HOME / "Library" / "LaunchAgents" / f"{LABEL}.plist"
STDOUT = HOME / ".local" / "log" / "hermes-nex-n2-sidecar" / "stdout.log"
STDERR = HOME / ".local" / "log" / "hermes-nex-n2-sidecar" / "stderr.log"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=False)


def main() -> int:
    STDOUT.parent.mkdir(parents=True, exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": LABEL,
        "ProgramArguments": [
            "/usr/bin/python3",
            str(REPO / "sidecar_proxy.py"),
            "--host",
            "127.0.0.1",
            "--port",
            "8092",
            "--upstream",
            "http://127.0.0.1:8090/v1",
        ],
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            "NEX_N2_DEFAULT_BUDGET": "512",
            "NEX_N2_DEEP_BUDGET": "2048",
            "NEX_N2_VISIBLE_OUTPUT_BUDGET": "2048",
            "NEX_N2_UPSTREAM_MAX_TOKENS": "4096",
            "NEX_N2_DEEP_CONCURRENCY": "1",
            "NEX_N2_REQUEST_TIMEOUT": "900",
        },
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(STDOUT),
        "StandardErrorPath": str(STDERR),
        "WorkingDirectory": str(REPO),
    }
    PLIST.write_bytes(plistlib.dumps(plist, sort_keys=False))
    uid = subprocess.check_output(["id", "-u"], text=True).strip()
    run(["launchctl", "bootout", f"gui/{uid}", str(PLIST)])
    subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", str(PLIST)], check=True)
    subprocess.run(["launchctl", "enable", f"gui/{uid}/{LABEL}"], check=False)
    subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{LABEL}"], check=True)
    print(PLIST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
