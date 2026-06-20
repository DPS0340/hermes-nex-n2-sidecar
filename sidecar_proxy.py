#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib import error, parse, request

LOG = logging.getLogger("hermes-nex-n2-sidecar")
CONTROL_PROFILE_KEYS = ("nex_n2_profile", "hermes_nex_n2_profile")
DEEP_SUFFIXES = (":deep", "#deep", "-deep")
DEEP_HEADER = "x-nex-n2-profile"
DEBUG_REWRITE_HEADER = "x-nex-n2-debug-rewrite"


@dataclass(frozen=True)
class SidecarConfig:
    upstream_base_url: str = "http://127.0.0.1:8090/v1"
    bind_host: str = "127.0.0.1"
    bind_port: int = 8092
    default_budget: int = 512
    deep_budget: int = 2048
    visible_output_budget: int = 2048
    upstream_max_tokens: int = 4096
    deep_concurrency: int = 1
    request_timeout: float = 900.0
    allow_origin: str = ""


class NexSidecarServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], RequestHandlerClass: type[BaseHTTPRequestHandler], cfg: SidecarConfig) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.cfg = cfg
        self.deep_semaphore = threading.BoundedSemaphore(max(1, cfg.deep_concurrency))


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _is_deep_request(payload: dict[str, Any]) -> bool:
    for key in CONTROL_PROFILE_KEYS:
        if str(payload.get(key, "")).lower() in {"deep", "2048", "true"}:
            return True
    model = str(payload.get("model", ""))
    if any(model.endswith(suffix) for suffix in DEEP_SUFFIXES):
        return True
    existing_budget = _int((payload.get("chat_template_kwargs") or {}).get("thinking_budget"), 0) if isinstance(payload.get("chat_template_kwargs"), dict) else 0
    return existing_budget > 512


def _normalize_model(model: Any) -> Any:
    if not isinstance(model, str):
        return model
    for suffix in DEEP_SUFFIXES:
        if model.endswith(suffix):
            return model[: -len(suffix)]
    return model


def prepare_chat_payload(payload: dict[str, Any], cfg: SidecarConfig) -> tuple[dict[str, Any], str]:
    """Apply Nex-N2 thinking policy before forwarding to vmlx."""
    rewritten: dict[str, Any] = dict(payload)
    mode = "deep" if _is_deep_request(rewritten) else "default"
    for key in CONTROL_PROFILE_KEYS:
        rewritten.pop(key, None)
    if "model" in rewritten:
        rewritten["model"] = _normalize_model(rewritten["model"])

    template_kwargs = rewritten.get("chat_template_kwargs")
    if not isinstance(template_kwargs, dict):
        template_kwargs = {}
    else:
        template_kwargs = dict(template_kwargs)

    requested_budget = _int(template_kwargs.get("thinking_budget"), 0)
    target_budget = cfg.deep_budget if mode == "deep" else cfg.default_budget
    if mode == "deep":
        budget = min(max(requested_budget, target_budget), cfg.deep_budget)
    else:
        budget = max(requested_budget, target_budget)
    template_kwargs["thinking_budget"] = budget
    rewritten["chat_template_kwargs"] = template_kwargs

    minimum_max_tokens = budget + max(0, cfg.visible_output_budget)
    current_max_tokens = _int(rewritten.get("max_tokens"), 0)
    rewritten["max_tokens"] = min(max(current_max_tokens, minimum_max_tokens), cfg.upstream_max_tokens)
    return rewritten, mode


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> NexSidecarServer:
        return cast(NexSidecarServer, self.server)

    def _send_bytes(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
        try:
            self.send_response(status)
            final_headers = headers or {}
            final_headers.setdefault("Content-Length", str(len(body)))
            final_headers.setdefault("Connection", "close")
            if self.app.cfg.allow_origin:
                final_headers.setdefault("Access-Control-Allow-Origin", self.app.cfg.allow_origin)
            for key, value in final_headers.items():
                lk = key.lower()
                if lk in {"transfer-encoding", "connection"}:
                    continue
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            LOG.info("client disconnected before response completed")

    def _json_error(self, status: int, message: str) -> None:
        payload = json.dumps({"error": {"message": message, "type": "sidecar_error"}}).encode()
        self._send_bytes(status, payload, {"Content-Type": "application/json"})

    def _target_url(self) -> str:
        path = self.path
        if path.startswith("/v1/"):
            suffix = path[len("/v1/") :]
        elif path == "/v1":
            suffix = ""
        else:
            suffix = path.lstrip("/")
        return self.app.cfg.upstream_base_url.rstrip("/") + "/" + suffix

    def _forward(self, method: str, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        upstream_headers = dict(headers or {})
        upstream_headers.pop("Host", None)
        req = request.Request(self._target_url(), data=body if method not in {"GET", "HEAD"} else None, headers=upstream_headers, method=method)
        try:
            with request.urlopen(req, timeout=self.app.cfg.request_timeout) as resp:
                data = resp.read()
                response_headers = {k: v for k, v in resp.headers.items() if k.lower() not in {"transfer-encoding", "connection"}}
                self._send_bytes(resp.status, data, response_headers)
        except error.HTTPError as exc:
            data = exc.read()
            response_headers = {k: v for k, v in exc.headers.items() if k.lower() not in {"transfer-encoding", "connection"}}
            self._send_bytes(exc.code, data, response_headers)
        except Exception as exc:
            LOG.exception("upstream request failed")
            self._json_error(502, f"upstream request failed: {exc}")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_bytes(204, b"", {"Allow": "GET,POST,OPTIONS", "Access-Control-Allow-Headers": "content-type,authorization,x-nex-n2-profile"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/healthz", "/health"}:
            self._send_bytes(200, b"ok\n", {"Content-Type": "text/plain"})
            return
        self._forward("GET", headers={k: v for k, v in self.headers.items()})

    def do_POST(self) -> None:  # noqa: N802
        length = _int(self.headers.get("Content-Length"), 0)
        body = self.rfile.read(length) if length else b""
        headers = {k: v for k, v in self.headers.items()}
        path = self.path.split("?", 1)[0]
        if path not in {"/v1/chat/completions", "/chat/completions", "/debug/rewrite"}:
            self._forward("POST", body=body, headers=headers)
            return
        try:
            payload = json.loads(body.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be object")
        except Exception as exc:
            self._json_error(400, f"invalid JSON body: {exc}")
            return
        header_profile = self.headers.get(DEEP_HEADER, "")
        if header_profile:
            payload = dict(payload)
            payload["nex_n2_profile"] = header_profile
        rewritten, mode = prepare_chat_payload(payload, self.app.cfg)
        if path == "/debug/rewrite" or str(self.headers.get(DEBUG_REWRITE_HEADER, "")).lower() in {"1", "true", "yes"}:
            safe = {k: v for k, v in rewritten.items() if k != "messages"}
            safe["message_count"] = len(rewritten.get("messages", [])) if isinstance(rewritten.get("messages"), list) else 0
            body = json.dumps({"mode": mode, "rewritten": safe}, separators=(",", ":")).encode("utf-8")
            self._send_bytes(200, body, {"Content-Type": "application/json"})
            return
        out = json.dumps(rewritten, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(out))
        if mode == "deep":
            with self.app.deep_semaphore:
                self._forward("POST", body=out, headers=headers)
        else:
            self._forward("POST", body=out, headers=headers)

    def log_message(self, format: str, *args: Any) -> None:
        LOG.info("%s - %s", self.address_string(), format % args)


def create_server(addr: tuple[str, int], cfg: SidecarConfig) -> NexSidecarServer:
    return NexSidecarServer(addr, ProxyHandler, cfg)


def config_from_env() -> SidecarConfig:
    return SidecarConfig(
        upstream_base_url=os.getenv("NEX_N2_UPSTREAM_BASE_URL", SidecarConfig.upstream_base_url),
        bind_host=os.getenv("NEX_N2_BIND_HOST", SidecarConfig.bind_host),
        bind_port=_int(os.getenv("NEX_N2_BIND_PORT"), SidecarConfig.bind_port),
        default_budget=_int(os.getenv("NEX_N2_DEFAULT_BUDGET"), SidecarConfig.default_budget),
        deep_budget=_int(os.getenv("NEX_N2_DEEP_BUDGET"), SidecarConfig.deep_budget),
        visible_output_budget=_int(os.getenv("NEX_N2_VISIBLE_OUTPUT_BUDGET"), SidecarConfig.visible_output_budget),
        upstream_max_tokens=_int(os.getenv("NEX_N2_UPSTREAM_MAX_TOKENS"), SidecarConfig.upstream_max_tokens),
        deep_concurrency=_int(os.getenv("NEX_N2_DEEP_CONCURRENCY"), SidecarConfig.deep_concurrency),
        request_timeout=float(os.getenv("NEX_N2_REQUEST_TIMEOUT", str(SidecarConfig.request_timeout))),
        allow_origin=os.getenv("NEX_N2_ALLOW_ORIGIN", ""),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes Nex-N2 local vmlx sidecar proxy")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--upstream", default=None)
    parser.add_argument("--log-level", default=os.getenv("NEX_N2_LOG_LEVEL", "INFO"))
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = config_from_env()
    if args.host or args.port or args.upstream:
        cfg = SidecarConfig(**{**cfg.__dict__, "bind_host": args.host or cfg.bind_host, "bind_port": args.port or cfg.bind_port, "upstream_base_url": args.upstream or cfg.upstream_base_url})
    server = create_server((cfg.bind_host, cfg.bind_port), cfg)
    stop = threading.Event()

    def handle_signal(signum: int, _frame: Any) -> None:
        LOG.info("received signal %s", signum)
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    LOG.info("listening on http://%s:%s/v1 -> %s", cfg.bind_host, cfg.bind_port, cfg.upstream_base_url)
    LOG.info("policy default_budget=%s deep_budget=%s visible_output_budget=%s max_tokens_cap=%s deep_concurrency=%s", cfg.default_budget, cfg.deep_budget, cfg.visible_output_budget, cfg.upstream_max_tokens, cfg.deep_concurrency)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
