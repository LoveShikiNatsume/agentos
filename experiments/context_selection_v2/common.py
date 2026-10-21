"""Versioned protocol; reuse only stable v1 utilities."""
from pathlib import Path
from experiments.context_selection.common import (digest, load, write_new, questions_for,
    model_config, add_model_arguments, config_from_args)

HERE = Path(__file__).resolve().parent
PROTOCOL_VERSION = "context-selection-v2.3"
# 题集版本不随探测后修正改变；manifest 仍为第二版冻结题集。
MANIFEST_VERSION = "context-selection-v2"
# 探测后登记：所有角色用后端 JSON 约束解码（vLLM response_format=json_object），各组一致。
RESPONSE_FORMAT = {"type": "json_object"}
ARMS = ("all", "declared", "declared_append", "bm25", "oracle", "bm25_append")
APPEND_LIMITS = {"max_requests": 2, "max_refs": 3}
READER_MAX_CALLS = 3
