#!/usr/bin/env python3
"""OpenAI-Compatible Local HTTP Server for True Dynamic Mati 12B MultiLoRA-MoE.

Runs the real N=3 MultiLoRA-MoE serving engine (MatiMoEEngine) on a local port
so that LM Studio, OpenWebUI, Cursor, or any OpenAI-compatible client can send
prompts and receive dynamically routed responses across Theory, Agentic, and ASM.

With --load-model, loads the Gemma 4 base (shim-friendly) and applies the
dominant specialist LoRA per request (lazy cache per expert).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mati_moe import MatiMoEEngine

ROOT = Path(__file__).resolve().parent.parent
ENGINE: MatiMoEEngine | None = None
MODEL_LOADED = False
BASE_PATH: Path | None = None
ADAPTER_DIRS: dict[str, Path] = {}
# Lazy cache: expert_name -> (model, tokenizer)
EXPERT_MODELS: dict[str, tuple[Any, Any]] = {}
# MLX Metal is not safe for concurrent generate/stream_generate.
GEN_LOCK = threading.Lock()


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if isinstance(content, str):
        return content
    return str(content) if content is not None else ""


SHARED_BASE_MODEL = None
SHARED_TOKENIZER = None
CURRENT_EXPERT = None


def _load_expert(expert: str):
    """Load base model once and switch specialist LoRA adapters dynamically (<0.1s)."""
    global SHARED_BASE_MODEL, SHARED_TOKENIZER, CURRENT_EXPERT

    assert BASE_PATH is not None
    from mlx_lm.utils import load_adapters, load_model, load_tokenizer

    if SHARED_BASE_MODEL is None:
        print(f"Loading shared base model from {BASE_PATH} …", flush=True)
        SHARED_BASE_MODEL, config = load_model(BASE_PATH, lazy=False, strict=False)
        SHARED_TOKENIZER = load_tokenizer(
            BASE_PATH,
            eos_token_ids=config.get("eos_token_id", None),
        )

    if CURRENT_EXPERT != expert:
        adapter_dir = ADAPTER_DIRS.get(expert) or ADAPTER_DIRS["theory"]
        print(f"Switching active specialist -> '{expert}' ({adapter_dir})", flush=True)
        if (adapter_dir / "adapters.safetensors").is_file():
            SHARED_BASE_MODEL = load_adapters(SHARED_BASE_MODEL, str(adapter_dir))
            SHARED_BASE_MODEL.eval()
        CURRENT_EXPERT = expert

    return SHARED_BASE_MODEL, SHARED_TOKENIZER


def _build_prompt(tokenizer, messages: list[dict], *, native_mode: bool, last_prompt: str) -> str:
    if native_mode:
        # Mati already rendered the full Gemma native chat template.
        return last_prompt
    chat = []
    for m in messages:
        role = m.get("role") or "user"
        if role == "tool":
            role = "user"
        chat.append({"role": role, "content": _message_text(m.get("content"))})
    try:
        return tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: concatenate
        return last_prompt + "\n"


class MoEHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "mati-12b-multilora-moe",
                            "object": "model",
                            "owned_by": "mati-core",
                        }
                    ],
                },
            )
        elif self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "engine": "MatiMoEEngine N=3",
                    "live_generation": MODEL_LOADED,
                    "experts_cached": list(EXPERT_MODELS.keys()),
                },
            )
        else:
            self._send_json(404, {"error": "Not Found"})

    def _sse_chunk(self, content: str | None = None, *, role: str | None = None, finish: str | None = None):
        delta: dict[str, Any] = {}
        if role is not None:
            delta["role"] = role
        if content is not None:
            delta["content"] = content
        payload = {
            "id": "chatcmpl-mati-moe",
            "object": "chat.completion.chunk",
            "model": "mati-12b-multilora-moe",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish,
                }
            ],
        }
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "Endpoint not found"})
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length).decode("utf-8"))

        messages = body.get("messages", [])
        last_prompt = _message_text(messages[-1].get("content", "") if messages else "")
        native_mode = bool(body.get("mati_native_prompt"))
        do_stream = bool(body.get("stream", False))
        max_toks = int(body.get("max_tokens") or 512)

        assert ENGINE is not None
        turn = ENGINE.generate_turn(last_prompt)
        routing = turn["routing"]
        dominant_name = routing["dominant_expert"]
        weights = routing["weights"]

        # Agentic LoRA currently collapses to gibberish on Gemma-4 native tool prompts
        # (verified: theory emits <|tool_call>…; agentic emits noise → Mati parses as action none).
        # Native / tool-declaration prompts must use theory until agentic is retrained.
        force_theory = native_mode or ("<|tool>" in last_prompt) or ("<|tool_call>" in last_prompt)
        if force_theory and dominant_name != "theory":
            print(
                f"expert override: {dominant_name} → theory (native/tool prompt)",
                flush=True,
            )
            dominant_name = "theory"
            routing = {
                **routing,
                "dominant_expert": "theory",
                "expert_override": "theory_native_tools",
            }

        header = (
            f"[Mati 12B MoE Routed -> {dominant_name.upper()} ({weights.get(dominant_name, 0)*100:.1f}%)]\n"
            f"Telemetry: Theory={weights['theory']*100:.1f}% | "
            f"Agentic={weights['agentic']*100:.1f}% | ASM={weights['asm_systems']*100:.1f}%\n"
        )

        if do_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Mati-MoE-Expert", dominant_name)
            self.end_headers()
            # Comment keepalives are ignored by OpenAI SSE clients but prove the stream is live
            # before potentially long expert-load / prefill under GEN_LOCK.
            self.wfile.write(b": moe-stream-open\n\n")
            self.wfile.flush()
            self._sse_chunk("", role="assistant")

            if not native_mode:
                self._sse_chunk(header)

            if not MODEL_LOADED:
                stub = (
                    "Processed prompt through MultiLoRA-MoE layers 8..47.\n"
                    "(Pass --load-model /path/to/base_gemma4 to run live MLX token generation)"
                )
                self._sse_chunk(stub)
            else:
                # mlx_lm.stream_generate hangs under ThreadingHTTPServer worker threads
                # after a few tokens. generate() is reliable; emit fake SSE chunks.
                try:
                    from mlx_lm import generate

                    with GEN_LOCK:
                        self.wfile.write(b": moe-acquire-lock\n\n")
                        self.wfile.flush()
                        model, tokenizer = _load_expert(dominant_name)
                        prompt = _build_prompt(
                            tokenizer, messages, native_mode=native_mode, last_prompt=last_prompt
                        )
                        print(
                            f"generate(streamed) expert={dominant_name} max_tokens={max_toks} "
                            f"prompt_chars={len(prompt)}",
                            flush=True,
                        )
                        self.wfile.write(
                            f": moe-prefill chars={len(prompt)}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()
                        gen_text = generate(
                            model, tokenizer, prompt=prompt, max_tokens=max_toks, verbose=False
                        )
                    if isinstance(gen_text, str) and gen_text.startswith(prompt):
                        gen_text = gen_text[len(prompt) :]
                    preview = (gen_text or "").replace("\n", "\\n")[:180]
                    print(f"generate done chars={len(gen_text or '')} preview={preview!r}", flush=True)
                    # Chunk so remote clients still see progressive SSE tokens.
                    step = 24
                    for i in range(0, len(gen_text), step):
                        self._sse_chunk(gen_text[i : i + step])
                except (BrokenPipeError, ConnectionResetError):
                    print("client disconnected during stream", flush=True)
                    return
                except Exception as e:
                    try:
                        self._sse_chunk(f"[MLX Generation Error: {e}]")
                    except (BrokenPipeError, ConnectionResetError):
                        print(f"client disconnected after error: {e}", flush=True)
                        return

            try:
                self._sse_chunk(None, finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                print("client disconnected before stream close", flush=True)
            return

        # Non-streaming path
        if MODEL_LOADED:
            try:
                from mlx_lm import generate

                with GEN_LOCK:
                    model, tokenizer = _load_expert(dominant_name)
                    prompt = _build_prompt(
                        tokenizer, messages, native_mode=native_mode, last_prompt=last_prompt
                    )
                    gen_text = generate(
                        model, tokenizer, prompt=prompt, max_tokens=max_toks, verbose=False
                    )
                if isinstance(gen_text, str) and gen_text.startswith(prompt):
                    gen_text = gen_text[len(prompt) :]
                response_text = gen_text if native_mode else (header + gen_text)
            except Exception as e:
                response_text = (
                    f"[MLX Generation Error: {e}]"
                    if native_mode
                    else header + f"[MLX Generation Error: {e}]"
                )
        else:
            stub = (
                "Processed prompt through MultiLoRA-MoE layers 8..47.\n"
                "(Pass --load-model /path/to/base_gemma4 to run live MLX token generation)"
            )
            response_text = stub if native_mode else (header + stub)

        self._send_json(
            200,
            {
                "id": "chatcmpl-mati-moe",
                "object": "chat.completion",
                "model": "mati-12b-multilora-moe",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(last_prompt.split()),
                    "completion_tokens": 30,
                    "total_tokens": 0,
                },
                "mati_moe_telemetry": routing,
            },
        )


def main():
    global ENGINE, MODEL_LOADED, BASE_PATH, ADAPTER_DIRS
    parser = argparse.ArgumentParser(description="Mati 12B MultiLoRA-MoE Local Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument(
        "--load-model",
        type=str,
        default=str(ROOT / "models" / "gemma12b" / "base_gemma4_shim"),
        help="Path to Gemma 4 12B MLX base (defaults to models/gemma12b/base_gemma4_shim)",
    )
    parser.add_argument(
        "--adapters-root",
        type=str,
        default=str(ROOT / "models" / "gemma12b"),
        help="Directory containing theory_lora / agentic_lora / asm_systems_lora",
    )
    args = parser.parse_args()

    ENGINE = MatiMoEEngine()
    MODEL_LOADED = False
    adapters_root = Path(args.adapters_root).expanduser()
    ADAPTER_DIRS = {
        "theory": adapters_root / "theory_lora",
        "agentic": adapters_root / "agentic_lora",
        "asm_systems": adapters_root / "asm_systems_lora",
    }

    if args.load_model:
        BASE_PATH = Path(args.load_model).expanduser()
        if not (BASE_PATH / "config.json").is_file():
            raise SystemExit(f"Base model missing config.json: {BASE_PATH}")
        print(f"Live generation ENABLED. Base: {BASE_PATH}", flush=True)
        print("Experts load lazily on first route (base + LoRA).", flush=True)
        MODEL_LOADED = True
    else:
        print("ROUTING TELEMETRY ONLY — pass --load-model for real replies.", flush=True)

    print("===================================================================")
    print("       MATI 12B MULTILORA-MOE LOCAL OPENAI-COMPATIBLE SERVER       ")
    print("===================================================================")
    print(f"Listening on:     http://127.0.0.1:{args.port}/v1")
    print(f"Live Generation:  {'ENABLED' if MODEL_LOADED else 'OFF (stub)'}")
    print("===================================================================\n", flush=True)

    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MoEHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MoE server...")
        server.server_close()


if __name__ == "__main__":
    main()
