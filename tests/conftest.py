import sys
from pathlib import Path

# Script helpers live alongside the installable source. Installed agentos keeps priority.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.append(str(SOURCE_ROOT))

import pytest

from agentos.llm.mock import MockLLMBackend

# Repository-shaped verifier inputs are data, not pytest test modules.
# The contest package keeps fixed inputs and archived evidence under the
# required Chinese-named material directory. They are data, not pytest
# modules, so do not collect their sample ``test_*.py`` files.
collect_ignore = [
    "测试及验证材料/fixtures",
    "测试及验证材料/evidence",
    "测试及验证材料/results",
]


@pytest.fixture
def mock_llm():
    return MockLLMBackend(responses={"default": "mock response"})
