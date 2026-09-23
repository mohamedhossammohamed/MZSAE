"""
server.py
High-Performance Local OpenAI-Compatible Server & Web Chat UI for BonsAI Models.
Powered by MSMZSAE Fused Metal Kernel on Apple Silicon.

Endpoints:
- GET  /health: Health check endpoint
- GET  /v1/models: Lists active models (bonsai-1.7b, bonsai-27b)
- POST /v1/chat/completions: OpenAI-compatible chat completion endpoint (streaming SSE & JSON)
- GET  /: Real-time interactive Web Chat UI
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Prevent package directory from shadowing standard library modules (e.g. logging)
_pkg_dir = str(Path(__file__).resolve().parent)
while _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)

REPO_DIR = Path(__file__).resolve().parent.parent.parent
if str(REPO_DIR / "src") not in sys.path:
    sys.path.insert(0, str(REPO_DIR / "src"))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(1, str(REPO_DIR))

import json
import time
import secrets
import argparse
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Any, Union

from mzsae.models.bonsai_runner import Bonsai1BRunner, Bonsai27BVisionRunner, BONSAI_2_27B_DIR

PORT = int(os.environ.get("PORT", 8000))
MODEL_1B_PATH = REPO_DIR / "models" / "bonsai-1.7b" / "Ternary-Bonsai-1.7B-PQ2_0.gguf"
TOK_1B_PATH = REPO_DIR / "models" / "bonsai-1.7b" / "tokenizer.json"

API_KEY = os.environ.get("BONSAI_API_KEY", "sk-bonsai-local")

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BonsAI v2 27B · Multimodal Vision & Inference</title>
    <style>
        :root {
            --bg: #0d1117;
            --panel: #161b22;
            --border: #30363d;
            --text: #f0f6fc;
            --muted: #8b949e;
            --accent: #58a6ff;
            --green: #3fb950;
            --purple: #bc8cff;
            --user-bg: #1f6feb;
            --bot-bg: #21262d;
            --code-bg: #090d13;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        header {
            background: var(--panel);
            border-bottom: 1px solid var(--border);
            padding: 14px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand h1 {
            font-size: 1.15rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        .badge {
            background: #238636;
            color: #fff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 7px;
            border-radius: 10px;
        }
        .badge.vision {
            background: var(--purple);
            color: #fff;
        }
        .meta {
            font-size: 0.82rem;
            color: var(--muted);
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .message {
            max-width: 80%;
            padding: 14px 18px;
            border-radius: 10px;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
            font-size: 0.95rem;
        }
        .message.user {
            align-self: flex-end;
            background: var(--user-bg);
            color: #fff;
            border-bottom-right-radius: 2px;
        }
        .message.assistant {
            align-self: flex-start;
            background: var(--bot-bg);
            border: 1px solid var(--border);
            border-bottom-left-radius: 2px;
        }
        .message .stats {
            margin-top: 8px;
            font-size: 0.75rem;
            color: var(--muted);
            border-top: 1px solid var(--border);
            padding-top: 6px;
            display: flex;
            gap: 12px;
        }
        .msg-image {
            max-width: 320px;
            max-height: 240px;
            border-radius: 8px;
            margin-bottom: 8px;
            display: block;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        #bottom-panel {
            background: var(--panel);
            border-top: 1px solid var(--border);
            display: flex;
            flex-direction: column;
        }
        #preview-tray {
            display: none;
            align-items: center;
            gap: 12px;
            padding: 10px 24px;
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid var(--border);
        }
        #preview-thumb {
            width: 48px;
            height: 48px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        #preview-name {
            font-size: 0.85rem;
            color: var(--muted);
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .remove-btn {
            background: transparent;
            color: var(--muted);
            border: none;
            font-size: 1.1rem;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .remove-btn:hover {
            color: #f85149;
            background: rgba(248, 81, 73, 0.1);
        }
        #input-panel {
            padding: 14px 24px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        textarea {
            flex: 1;
            background: var(--bg);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 12px 16px;
            border-radius: 8px;
            resize: none;
            font-family: inherit;
            font-size: 0.95rem;
            height: 48px;
            outline: none;
            transition: border-color 0.2s;
        }
        textarea:focus {
            border-color: var(--accent);
        }
        button.action-btn {
            background: var(--accent);
            color: #0d1117;
            border: none;
            padding: 0 22px;
            height: 48px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            transition: opacity 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        button.attach-btn {
            background: #21262d;
            color: var(--text);
            border: 1px solid var(--border);
            padding: 0 16px;
            height: 48px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 0.9rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: background 0.2s;
        }
        button.attach-btn:hover {
            background: #30363d;
        }
        button:hover {
            opacity: 0.9;
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <h1>BonsAI v2 27B</h1>
            <span class="badge">MLX 2-BIT METAL</span>
            <span class="badge vision">VISION</span>
        </div>
        <div class="meta">
            Hardware: Apple Silicon M4 · Hybrid (48 SSM + 16 GQA) · <code>/v1/chat/completions</code>
        </div>
    </header>

    <div id="chat-container">
        <div class="message assistant">
            Hello! I am BonsAI v2 27B running locally on Apple Silicon Metal with hybrid SSM-GQA architecture and full multimodal vision support. How can I assist you today?
        </div>
    </div>

    <div id="bottom-panel">
        <div id="preview-tray">
            <img id="preview-thumb" src="" alt="preview" />
            <span id="preview-name">image.png</span>
            <button class="remove-btn" onclick="clearAttachedImage()" title="Remove image">✕</button>
        </div>
        <div id="input-panel">
            <input type="file" id="image-file-input" accept="image/*" style="display:none" onchange="handleImageSelected(event)" />
            <button class="attach-btn" type="button" onclick="document.getElementById('image-file-input').click()" title="Attach image for vision analysis">
                📷 Attach
            </button>
            <textarea id="prompt-input" placeholder="Type a message or prompt... (Press Enter to send, Shift+Enter for newline)"></textarea>
            <button id="send-btn" class="action-btn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const promptInput = document.getElementById('prompt-input');
        const sendBtn = document.getElementById('send-btn');
        const fileInput = document.getElementById('image-file-input');
        const previewTray = document.getElementById('preview-tray');
        const previewThumb = document.getElementById('preview-thumb');
        const previewName = document.getElementById('preview-name');

        let conversationHistory = [];
        let attachedImageBase64 = null;

        function handleImageSelected(e) {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(evt) {
                attachedImageBase64 = evt.target.result;
                previewThumb.src = attachedImageBase64;
                previewName.innerText = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
                previewTray.style.display = 'flex';
                promptInput.focus();
            };
            reader.readAsDataURL(file);
        }

        function clearAttachedImage() {
            attachedImageBase64 = null;
            fileInput.value = '';
            previewTray.style.display = 'none';
            previewThumb.src = '';
            previewName.innerText = '';
        }

        promptInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        async function sendMessage() {
            const text = promptInput.value.trim();
            if (!text && !attachedImageBase64) return;

            const sendImage = attachedImageBase64;
            clearAttachedImage();

            promptInput.value = '';
            promptInput.disabled = true;
            sendBtn.disabled = true;

            // Render user message in chat
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            if (sendImage) {
                const img = document.createElement('img');
                img.className = 'msg-image';
                img.src = sendImage;
                userMsg.appendChild(img);
            }
            if (text) {
                const span = document.createElement('span');
                span.innerText = text;
                userMsg.appendChild(span);
            }
            chatContainer.appendChild(userMsg);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            // Construct OpenAI multimodal message structure
            let msgContent;
            if (sendImage) {
                msgContent = [
                    { type: 'text', text: text || 'Please describe and analyze this image.' },
                    { type: 'image_url', image_url: { url: sendImage } }
                ];
            } else {
                msgContent = text;
            }
            conversationHistory.push({ role: 'user', content: msgContent });

            // Append assistant placeholder
            const assistantMsg = document.createElement('div');
            assistantMsg.className = 'message assistant';
            assistantMsg.innerText = 'Thinking...';
            chatContainer.appendChild(assistantMsg);
            chatContainer.scrollTop = chatContainer.scrollHeight;

            const t0 = performance.now();
            let accumulated = '';

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': 'Bearer sk-bonsai-local'
                    },
                    body: JSON.stringify({
                        model: 'bonsai-2-27b',
                        messages: conversationHistory,
                        max_tokens: 1024,
                        temperature: 0.7,
                        stream: true
                    })
                });

                if (!res.ok) {
                    throw new Error(`Server returned ${res.status}: ${await res.text()}`);
                }

                assistantMsg.innerText = '';
                const reader = res.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || !trimmed.startsWith('data: ')) continue;
                        const dataStr = trimmed.substring(6);
                        if (dataStr === '[DONE]') break;

                        try {
                            const parsed = JSON.parse(dataStr);
                            const delta = parsed.choices?.[0]?.delta?.content || '';
                            if (delta) {
                                accumulated += delta;
                                assistantMsg.innerText = accumulated;
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }
                        } catch (err) {}
                    }
                }

                const elapsedMs = performance.now() - t0;
                conversationHistory.push({ role: 'assistant', content: accumulated });

                const statsDiv = document.createElement('div');
                statsDiv.className = 'stats';
                statsDiv.innerHTML = `<span>Latency: ${elapsedMs.toFixed(0)} ms</span><span>Engine: BonsAI v2 27B (MLX Metal)</span>`;
                assistantMsg.appendChild(statsDiv);

            } catch (err) {
                assistantMsg.innerText = 'Error: ' + err.message;
            } finally {
                promptInput.disabled = false;
                sendBtn.disabled = false;
                promptInput.focus();
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
    </script>
</body>
</html>
"""


class BonsaiRequestHandler(BaseHTTPRequestHandler):
    server: BonsaiServer

    def _send_json(self, status: int, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            html = CHAT_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html)
            return

        if parsed.path == "/health":
            active_model = getattr(self.server.runner, "model_id", "bonsai-2-27b")
            self._send_json(200, {
                "status": "ok",
                "backend": "MLX / MSMZSAE Fused Metal (2-bit Hadamard affine)" if "27b" in active_model else "MSMZSAE Fused Metal Kernel",
                "models": ["bonsai-2-27b", "bonsai-27b", "bonsai-1.7b"],
                "active_model": active_model,
                "time": time.time()
            })
            return

        if parsed.path in ("/v1/models", "/models"):
            self._send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": "bonsai-2-27b",
                        "object": "model",
                        "created": 1742700000,
                        "owned_by": "prism-ml",
                        "permission": []
                    },
                    {
                        "id": "bonsai-27b",
                        "object": "model",
                        "created": 1742700000,
                        "owned_by": "prism-ml",
                        "permission": []
                    },
                    {
                        "id": "bonsai-1.7b",
                        "object": "model",
                        "created": 1742700000,
                        "owned_by": "bonsai",
                        "permission": []
                    }
                ]
            })
            return

        self.send_error(404, f"Path not found: {parsed.path}")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path in ("/v1/chat/completions", "/chat/completions"):
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)

            try:
                data = json.loads(body_bytes.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": {"message": f"Invalid JSON payload: {e}"}})
                return

            messages = data.get("messages", [])
            model_id = data.get("model", getattr(self.server.runner, "model_id", "bonsai-2-27b"))
            max_tokens = min(data.get("max_tokens", 512), 2048)
            temperature = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.85))
            top_k = int(data.get("top_k", 20))
            stream = bool(data.get("stream", False))

            chat_id = f"chatcmpl-{secrets.token_hex(12)}"
            created_ts = int(time.time())

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                try:
                    for chunk in self.server.runner.generate_stream(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k
                    ):
                        chunk_obj = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }
                            ]
                        }
                        self.wfile.write(f"data: {json.dumps(chunk_obj)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    done_obj = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(done_obj)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            else:
                response_text = self.server.runner.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k
                )

                resp_obj = {
                    "id": chat_id,
                    "object": "chat.completion",
                    "created": created_ts,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(str(messages)),
                        "completion_tokens": len(response_text.split()),
                        "total_tokens": len(str(messages)) + len(response_text.split())
                    }
                }
                self._send_json(200, resp_obj)
                return

        self.send_error(404, f"Endpoint not found: {parsed.path}")


class BonsaiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, runner: Any):
        super().__init__(server_address, handler_class)
        self.runner = runner


def start_server(port: int = PORT, model_choice: str = "bonsai-2-27b"):
    print("=" * 78)
    print("STARTING BONSAI LOCAL OPENAI-COMPATIBLE INFERENCE SERVER")
    print("Backend: MLX / MSMZSAE Hybrid Architecture on Apple Silicon Unified Memory")
    print("=" * 78)

    if model_choice in ("bonsai-2-27b", "bonsai-27b", "27b"):
        print(f"[*] Initializing model: {model_choice} (v2 Multimodal Vision)...")
        runner = Bonsai27BVisionRunner(model_dir=BONSAI_2_27B_DIR)
    else:
        print(f"[*] Initializing model: {model_choice} (1.7B Language)...")
        runner = Bonsai1BRunner(
            model_path=MODEL_1B_PATH,
            tokenizer_path=TOK_1B_PATH
        )

    server = BonsaiServer(("127.0.0.1", port), BonsaiRequestHandler, runner=runner)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print(f"\n[✓] Server running on http://127.0.0.1:{port}")
    print(f"  • Web Chat UI:        http://127.0.0.1:{port}/")
    print(f"  • Health Endpoint:    http://127.0.0.1:{port}/health")
    print(f"  • Models Endpoint:    http://127.0.0.1:{port}/v1/models")
    print(f"  • Chat Completions:   http://127.0.0.1:{port}/v1/chat/completions")
    print(f"  • Active Model:       {runner.model_id}")
    print(f"  • API Key:            sk-bonsai-local\n")

    try:
        runner.process_loop()
    except KeyboardInterrupt:
        print("\n[*] Server stopping...")
    finally:
        runner.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BonsAI Local Server")
    parser.add_argument("--port", type=int, default=PORT, help="Port to bind server")
    parser.add_argument("--model", type=str, default="bonsai-2-27b",
                        choices=["bonsai-2-27b", "bonsai-27b", "bonsai-1.7b"],
                        help="Model to serve (default: bonsai-2-27b)")
    args = parser.parse_args()
    start_server(port=args.port, model_choice=args.model)
