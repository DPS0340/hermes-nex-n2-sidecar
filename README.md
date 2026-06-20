# hermes-nex-n2-sidecar

Durable local sidecar for Hermes + vmlx Nex-N2-mini on Mac Studio M4 Max / 64GB.

Goal: keep Hermes upstream untouched. Hermes points at this OpenAI-compatible proxy; proxy forwards to vmlx on `127.0.0.1:8090/v1` and injects safe Nex-N2 thinking policy.

## Policy

- Default requests: `chat_template_kwargs.thinking_budget = 512`
- Deep requests: `chat_template_kwargs.thinking_budget = 2048`
- Deep concurrency: bounded by semaphore, default `1`
- Output cap: `max_tokens >= thinking_budget + visible_output_budget`, capped by upstream server limit
- `/v1/models`: passthrough
- No API keys, no cloud fallback, no Hermes source patch

## Deep request selectors

Any of these trigger deep profile:

- model suffix: `mlx-community/Nex-N2-mini-nvfp4:deep`
- JSON control field: `"nex_n2_profile": "deep"` (stripped before upstream)
- HTTP header: `X-Nex-N2-Profile: deep`
- incoming `chat_template_kwargs.thinking_budget > 512`
- **Context-aware** (body or headers): `context_used_tokens / context_window >= context_deep_ratio` AND `hard_task_score >= context_deep_score` — only when Hermes sends metadata; never auto-triggers from conversation length alone

## Run foreground

```bash
python3 sidecar_proxy.py --port 8092 --upstream http://127.0.0.1:8090/v1
```

## Test

```bash
python3 -m unittest -v tests/test_sidecar_proxy.py
```

## Environment

```text
NEX_N2_UPSTREAM_BASE_URL=http://127.0.0.1:8090/v1
NEX_N2_BIND_HOST=127.0.0.1
NEX_N2_BIND_PORT=8092
NEX_N2_DEFAULT_BUDGET=512
NEX_N2_DEEP_BUDGET=2048
NEX_N2_VISIBLE_OUTPUT_BUDGET=2048
NEX_N2_UPSTREAM_MAX_TOKENS=4096
NEX_N2_DEEP_CONCURRENCY=1
NEX_N2_REQUEST_TIMEOUT=900
NEX_N2_CONTEXT_DEEP_RATIO=0.75
NEX_N2_CONTEXT_DEEP_SCORE=2
```

## launchd

Template lives in `launchd/ai.hermes.nex-n2-sidecar.plist`.
Install script writes it to `~/Library/LaunchAgents/ai.hermes.nex-n2-sidecar.plist` and bootstraps it.
