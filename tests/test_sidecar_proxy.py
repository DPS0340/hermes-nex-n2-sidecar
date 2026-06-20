from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sidecar_proxy import SidecarConfig, create_server, prepare_chat_payload


class TestPayloadRewrite(unittest.TestCase):
    def test_default_budget_is_added_and_max_tokens_has_visible_room(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096)
        body = {"model": "mlx-community/Nex-N2-mini-nvfp4", "messages": [], "max_tokens": 1000}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "default")
        self.assertEqual(rewritten["model"], "mlx-community/Nex-N2-mini-nvfp4")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 512)
        self.assertEqual(rewritten["max_tokens"], 2560)

    def test_deep_profile_sets_2048_budget_and_strips_control_field(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096)
        body = {"model": "mlx-community/Nex-N2-mini-nvfp4", "messages": [], "nex_n2_profile": "deep"}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "deep")
        self.assertNotIn("nex_n2_profile", rewritten)
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 2048)
        self.assertEqual(rewritten["max_tokens"], 4096)

    def test_model_suffix_deep_is_normalized_for_upstream(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096)
        body = {"model": "mlx-community/Nex-N2-mini-nvfp4:deep", "messages": []}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "deep")
        self.assertEqual(rewritten["model"], "mlx-community/Nex-N2-mini-nvfp4")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 2048)

    def test_existing_higher_budget_is_preserved_and_deep_limited(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=512, upstream_max_tokens=4096)
        body = {"model": "m", "messages": [], "chat_template_kwargs": {"thinking_budget": 4096}, "max_tokens": 1}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "deep")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 2048)
        self.assertEqual(rewritten["max_tokens"], 2560)

    def test_context_body_triggers_deep_when_ratio_and_score_high(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096, context_deep_ratio=0.70, context_deep_score=2)
        body = {"model": "m", "messages": [], "nex_n2_context": {"used_tokens": 50000, "context_window": 65536, "hard_task_score": 3}}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "deep")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 2048)
        self.assertNotIn("nex_n2_context", rewritten)

    def test_context_body_does_not_trigger_deep_when_ratio_low(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096, context_deep_ratio=0.70, context_deep_score=2)
        body = {"model": "m", "messages": [], "nex_n2_context": {"used_tokens": 10000, "context_window": 65536, "hard_task_score": 3}}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "default")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 512)
        self.assertNotIn("nex_n2_context", rewritten)

    def test_context_body_does_not_trigger_deep_when_score_low(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096, context_deep_ratio=0.70, context_deep_score=2)
        body = {"model": "m", "messages": [], "nex_n2_context": {"used_tokens": 50000, "context_window": 65536, "hard_task_score": 0}}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "default")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 512)
        self.assertNotIn("nex_n2_context", rewritten)

    def test_context_headers_trigger_deep(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096, context_deep_ratio=0.70, context_deep_score=2)
        body = {"model": "m", "messages": []}
        headers = {"x-nex-n2-context-used-tokens": "50000", "x-nex-n2-context-window": "65536", "x-nex-n2-hard-task-score": "3"}

        rewritten, mode = prepare_chat_payload(body, cfg, headers)

        self.assertEqual(mode, "deep")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 2048)

    def test_context_null_body_does_not_crash(self) -> None:
        cfg = SidecarConfig(default_budget=512, deep_budget=2048, visible_output_budget=2048, upstream_max_tokens=4096)
        body = {"model": "m", "messages": [], "nex_n2_context": None}

        rewritten, mode = prepare_chat_payload(body, cfg)

        self.assertEqual(mode, "default")
        self.assertEqual(rewritten["chat_template_kwargs"]["thinking_budget"], 512)
        self.assertNotIn("nex_n2_context", rewritten)


class RecordingUpstream(BaseHTTPRequestHandler):
    requests_seen: list[dict] = []
    active_deep = 0
    max_active_deep = 0
    lock = threading.Lock()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            payload = {"object": "list", "data": [{"id": "mlx-community/Nex-N2-mini-nvfp4", "object": "model"}]}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        is_deep = body.get("chat_template_kwargs", {}).get("thinking_budget") == 2048
        with self.lock:
            self.requests_seen.append(body)
            if is_deep:
                type(self).active_deep += 1
                type(self).max_active_deep = max(type(self).max_active_deep, type(self).active_deep)
        if is_deep:
            time.sleep(0.2)
        payload = {"id": "ok", "choices": [{"message": {"role": "assistant", "content": "OK"}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        with self.lock:
            if is_deep:
                type(self).active_deep -= 1

    def log_message(self, format: str, *args: object) -> None:
        return


class TestHTTPProxy(unittest.TestCase):
    def setUp(self) -> None:
        RecordingUpstream.requests_seen = []
        RecordingUpstream.active_deep = 0
        RecordingUpstream.max_active_deep = 0
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstream)
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        upstream_url = f"http://127.0.0.1:{self.upstream.server_port}/v1"
        cfg = SidecarConfig(upstream_base_url=upstream_url, deep_concurrency=1)
        self.proxy = create_server(("127.0.0.1", 0), cfg)
        self.proxy_thread = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.proxy_thread.start()
        self.base = f"http://127.0.0.1:{self.proxy.server_port}/v1"

    def tearDown(self) -> None:
        self.proxy.shutdown()
        self.upstream.shutdown()
        self.proxy.server_close()
        self.upstream.server_close()

    def test_models_passthrough(self) -> None:
        with request.urlopen(f"{self.base}/models", timeout=5) as resp:
            data = json.loads(resp.read())
        self.assertEqual(data["data"][0]["id"], "mlx-community/Nex-N2-mini-nvfp4")

    def test_debug_rewrite_reports_policy_without_forwarding_upstream(self) -> None:
        payload = json.dumps({"model": "mlx-community/Nex-N2-mini-nvfp4:deep", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}).encode()
        req = request.Request(f"http://127.0.0.1:{self.proxy.server_port}/debug/rewrite", data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        self.assertEqual(data["mode"], "deep")
        self.assertEqual(data["rewritten"]["model"], "mlx-community/Nex-N2-mini-nvfp4")
        self.assertEqual(data["rewritten"]["chat_template_kwargs"]["thinking_budget"], 2048)
        self.assertEqual(data["rewritten"]["max_tokens"], 4096)
        self.assertEqual(data["rewritten"]["message_count"], 1)
        self.assertEqual(RecordingUpstream.requests_seen, [])

    def test_deep_requests_are_serialized_by_semaphore(self) -> None:
        def post() -> None:
            payload = json.dumps({"model": "mlx-community/Nex-N2-mini-nvfp4:deep", "messages": []}).encode()
            req = request.Request(f"{self.base}/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST")
            with request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)

        threads = [threading.Thread(target=post) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(RecordingUpstream.requests_seen), 3)
        self.assertEqual(RecordingUpstream.max_active_deep, 1)
        self.assertTrue(all(r["chat_template_kwargs"]["thinking_budget"] == 2048 for r in RecordingUpstream.requests_seen))


if __name__ == "__main__":
    unittest.main()
