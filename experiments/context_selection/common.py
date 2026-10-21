"""Stable artifacts and explicit backend configuration."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "context-selection-v1"
ARMS = ("all", "declared", "bm25", "oracle")

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

def questions_for(manifest, split):
    return [q for q in manifest["questions"] if split == "all" or q["split"] == split]

def model_config(*, base_url=None, model=None, stub=False, seed=20261019, max_tokens=4096, enable_thinking=None):
    if max_tokens < 2048:
        raise ValueError("max_tokens must be at least 2048")
    if not stub and (not base_url or not model):
        raise ValueError("real backend requires explicit --base-url and --model")
    return dict(base_url=base_url or "", model=model or "context-stub",
                backend="stub" if stub else "openai", seed=seed, max_tokens=max_tokens, enable_thinking=enable_thinking,
                temperature=0, api_key_env="OPENAI_API_KEY", timeout=180, retries=0)

def add_model_arguments(parser):
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--stub", action="store_true", help="in-process fake backend; no HTTP")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default",
                        help="Qwen3 系列聊天模板的思考开关；default 不发送该参数")

def config_from_args(args):
    return model_config(base_url=args.base_url, model=args.model, stub=args.stub,
                        seed=args.seed, max_tokens=args.max_tokens,
                        enable_thinking={"default": None, "on": True, "off": False}[args.thinking])
