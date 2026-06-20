#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"PyYAML required: {exc}", file=sys.stderr)
    raise SystemExit(1)

CONFIG = Path.home() / ".hermes" / "config.yaml"
PROXY_URL = "http://127.0.0.1:8092/v1"
MODEL = "mlx-community/Nex-N2-mini-nvfp4"
DEEP_MODEL = "mlx-community/Nex-N2-mini-nvfp4:deep"


def main() -> int:
    data = yaml.safe_load(CONFIG.read_text()) or {}
    model = data.setdefault("model", {})
    model["provider"] = "custom"
    model["default"] = MODEL
    model["base_url"] = PROXY_URL
    model["max_tokens"] = 32768
    model["context_length"] = 65536
    model["api_key"] = ""
    # Local-only: remove stale cloud fallback fields.
    model.pop("fallback_provider", None)
    model.pop("fallback_model", None)

    providers = [p for p in data.get("custom_providers", []) if isinstance(p, dict)]
    providers = [p for p in providers if p.get("name") not in {"vmlx Nex-N2-mini sidecar (localhost:8092)", "vmlx Nex-N2-mini sidecar deep (localhost:8092)"}]
    providers.append({
        "name": "vmlx Nex-N2-mini sidecar (localhost:8092)",
        "base_url": PROXY_URL,
        "model": MODEL,
        "context_length": 65536,
        "extra_body": {"chat_template_kwargs": {"thinking_budget": 512}},
    })
    providers.append({
        "name": "vmlx Nex-N2-mini sidecar deep (localhost:8092)",
        "base_url": PROXY_URL,
        "model": DEEP_MODEL,
        "context_length": 65536,
        "extra_body": {"chat_template_kwargs": {"thinking_budget": 2048}, "max_tokens": 4096},
    })
    data["custom_providers"] = providers
    CONFIG.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    print("updated", CONFIG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
