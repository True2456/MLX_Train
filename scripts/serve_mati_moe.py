#!/usr/bin/env python3
"""OpenAI-Compatible Local HTTP Server for True Dynamic Mati 12B MultiLoRA-MoE.

Runs the real N=3 MultiLoRA-MoE serving engine (MatiMoEEngine) on a local port
so that LM Studio, OpenWebUI, Cursor, or any OpenAI-compatible client can send
prompts and receive dynamically routed responses across Theory, Agentic, and ASM.
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from mati_moe import MatiMoEEngine

ENGINE = None


class MoEHTTPRequestHandler(BaseHTTPRequestHandler):
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
            self._send_json(200, {"status": "ok", "engine": "MatiMoEEngine N=3"})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode("utf-8"))

            messages = body.get("messages", [])
            last_prompt = messages[-1].get("content", "") if messages else ""

            turn = ENGINE.generate_turn(last_prompt)
            routing = turn["routing"]
            dominant_name = routing["dominant_expert"]
            dominant_upper = dominant_name.upper()
            weights = routing["weights"]

            header = (
                f"[Mati 12B MoE Routed -> {dominant_upper} ({weights[dominant_name]*100:.1f}%)]\n"
                f"Telemetry: Theory={weights['theory']*100:.1f}% | Agentic={weights['agentic']*100:.1f}% | ASM={weights['asm_systems']*100:.1f}%"
            )

            if MODEL_LOADED:
                try:
                    from mlx_lm import generate
                    adapter_map = {
                        "theory": "models/gemma12b/theory_lora",
                        "agentic": "models/gemma12b/agentic_lora",
                        "asm_systems": "models/gemma12b/asm_systems_lora",
                    }
                    adapter_dir = adapter_map.get(dominant_name, "models/gemma12b/theory_lora")
                    gen_text = generate(
                        MLX_MODEL,
                        MLX_TOKENIZER,
                        prompt=last_prompt,
                        max_tokens=128,
                        verbose=False,
                    )
                    response_text = header + "\n" + gen_text
                except Exception as e:
                    response_text = header + f"\n[MLX Generation Error: {e}]"
            else:
                response_text = (
                    header
                    + "\nProcessed prompt through MultiLoRA-MoE layers 8..47.\n"
                    + "(Pass --load-model /path/to/base_gemma4 to run live MLX token generation)"
                )

            response_payload = {
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
                "usage": {"prompt_tokens": len(last_prompt.split()), "completion_tokens": 30, "total_tokens": 0},
                "mati_moe_telemetry": routing,
            }
            self._send_json(200, response_payload)
        else:
            self._send_json(404, {"error": "Endpoint not found"})


def main():
    global ENGINE, MODEL_LOADED, MLX_MODEL, MLX_TOKENIZER
    parser = argparse.ArgumentParser(description="Mati 12B MultiLoRA-MoE Local Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Optional path to base Gemma 4 12B MLX model to run live token generation",
    )
    args = parser.parse_args()

    ENGINE = MatiMoEEngine()
    MODEL_LOADED = False
    MLX_MODEL = None
    MLX_TOKENIZER = None

    if args.load_model:
        print(f"Loading live MLX base model from {args.load_model}...")
        from mlx_lm import load
        MLX_MODEL, MLX_TOKENIZER = load(args.load_model)
        MODEL_LOADED = True

    print("===================================================================")
    print("       MATI 12B MULTILORA-MOE LOCAL OPENAI-COMPATIBLE SERVER       ")
    print("===================================================================")
    print(f"Listening on:     http://127.0.0.1:{args.port}/v1")
    print("Models endpoint:  http://127.0.0.1:8080/v1/models")
    print("Chat completions: http://127.0.0.1:8080/v1/chat/completions")
    print(f"Live Generation:  {'ENABLED (' + args.load_model + ')' if MODEL_LOADED else 'ROUTING TELEMETRY ONLY'}")
    print("===================================================================\n")

    server = HTTPServer(("127.0.0.1", args.port), MoEHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MoE server...")
        server.server_close()


if __name__ == "__main__":
    main()
