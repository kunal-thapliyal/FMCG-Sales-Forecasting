"""
Phase 3 test suite — chat_tools.py (tool wrappers) and agent.py (LangGraph
orchestration), plus the /chat endpoint's degraded behavior when
GROQ_API_KEY is missing.

Run from the phase2/ directory:
    pytest tests/test_chat_tools.py -v

Requires the DuckDB warehouse to already be built:
    python warehouse/build_warehouse.py

No real Groq API key or network access is needed: chat_tools tests hit the
real local DuckDB/model stack (same as test_backend.py), and agent tests
monkeypatch agent._get_client with a fake Groq client so the LangGraph loop
itself is exercised without a live LLM call.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import chat_tools  # noqa: E402


# ---------------------------------------------------------------------------
# chat_tools.py — direct unit tests against the real warehouse
# ---------------------------------------------------------------------------

def test_get_metadata_shape():
    meta = chat_tools.get_metadata()
    assert "error" not in meta
    for key in ("skus", "regions", "channels", "categories", "brands", "segments", "pack_types"):
        assert isinstance(meta[key], list) and len(meta[key]) > 0
    assert meta["min_date"] < meta["max_date"]


def test_invalid_sku_rejected():
    result = chat_tools.get_sales_summary(skus=["NOT-A-REAL-SKU"])
    assert "error" in result
    assert "sku" in result["error"].lower()


def test_invalid_region_rejected():
    result = chat_tools.compare_sales(dimension="region", regions=["Mars"])
    assert "error" in result


def test_invalid_dimension_rejected():
    result = chat_tools.compare_sales(dimension="not_a_dimension")
    assert "error" in result


def test_valid_sales_summary():
    result = chat_tools.get_sales_summary(categories=["Juice"], start_date="2024-01-01", end_date="2024-12-31")
    assert "error" not in result
    assert result["n_rows"] > 0
    assert result["total_units_sold"] > 0


def test_compare_sales_ranks_categories():
    result = chat_tools.compare_sales(dimension="category", top_n=5)
    assert "error" not in result
    assert len(result["ranked"]) <= 5
    # Should be sorted descending by units_sold, matching analytics.get_comparison.
    values = [row["units_sold"] for row in result["ranked"]]
    assert values == sorted(values, reverse=True)


def test_inventory_analysis():
    result = chat_tools.get_inventory_analysis(dimension="region")
    assert "error" not in result
    assert len(result["groups"]) == 3  # PL-Central, PL-North, PL-South


def test_promotion_analysis():
    result = chat_tools.get_promotion_analysis(dimension="category")
    assert "error" not in result
    for row in result["groups"]:
        assert "uplift_pct" in row


def test_forecast_series_available():
    result = chat_tools.get_forecast_series()
    assert "error" not in result
    assert result["count"] > 0


def test_forecast_last_known_valid_series():
    series = chat_tools.get_forecast_series()["series"][0]
    result = chat_tools.get_forecast_last_known(series["sku"], series["region"], series["channel"])
    assert "error" not in result
    assert "price_unit" in result


def test_forecast_last_known_invalid_sku():
    result = chat_tools.get_forecast_last_known("NOT-A-REAL-SKU", "PL-Central", "Retail")
    assert "error" in result


def test_forecast_predict_missing_operational_inputs():
    """The tool should ask the LLM to call get_forecast_last_known first rather
    than silently guessing price/stock/etc."""
    series = chat_tools.get_forecast_series()["series"][0]
    result = chat_tools.forecast_predict(
        target_date="2025-06-01", sku=series["sku"], region=series["region"], channel=series["channel"],
    )
    assert "error" in result
    assert "get_forecast_last_known" in result["error"]


def test_forecast_predict_future_date_is_allowed():
    """CRITICAL: a target_date beyond the historical max_date must NOT be
    rejected — forecasting the future is the point of this tool."""
    series = chat_tools.get_forecast_series()["series"][0]
    last_known = chat_tools.get_forecast_last_known(series["sku"], series["region"], series["channel"])
    meta = chat_tools.get_metadata()
    assert meta["max_date"] < "2025-06-01"  # sanity: date really is in the future
    result = chat_tools.forecast_predict(
        target_date="2025-06-01", sku=series["sku"], region=series["region"], channel=series["channel"],
        price_unit=last_known["price_unit"], promotion_flag=last_known["promotion_flag"],
        delivery_days=last_known["delivery_days"], stock_available=last_known["stock_available"],
        delivered_qty=last_known["delivered_qty"], pack_type=last_known["pack_type"],
    )
    assert "error" not in result, f"future date was wrongly rejected: {result}"
    assert result["date"] == "2025-06-01"
    assert "predicted_units_sold" in result
    assert "forecast" in result["note"].lower()


def test_forecast_predict_invalid_sku_gives_domain_error_not_fabrication():
    result = chat_tools.forecast_predict(
        target_date="2025-01-01", sku="NOT-A-REAL-SKU", region="PL-Central", channel="Retail",
        price_unit=5.0, promotion_flag=False, delivery_days=3, stock_available=150, delivered_qty=170,
    )
    assert "error" in result
    assert "predicted_units_sold" not in result


def test_call_function_returns_json_string_never_raises():
    out = chat_tools.call_function("compare_sales", {"dimension": "category", "top_n": 3})
    parsed = json.loads(out)
    assert "ranked" in parsed

    out_bad_tool = chat_tools.call_function("this_tool_does_not_exist", {})
    assert "error" in json.loads(out_bad_tool)

    out_bad_args = chat_tools.call_function("compare_sales", {"nonexistent_kwarg": 1})
    assert "error" in json.loads(out_bad_args)


def test_tool_schemas_are_well_formed():
    """Every declared tool must be callable and have OpenAI/Groq-shaped schema."""
    tool_names = {t["function"]["name"] for t in chat_tools.TOOLS}
    assert tool_names == set(chat_tools.FUNCTIONS.keys())
    for t in chat_tools.TOOLS:
        assert t["type"] == "function"
        assert "description" in t["function"] and t["function"]["description"]
        assert "parameters" in t["function"]


# ---------------------------------------------------------------------------
# agent.py — LangGraph loop, mocked Groq client (no network / API key needed)
# ---------------------------------------------------------------------------

class _FakeToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeResponse:
    def __init__(self, message):
        self.choices = [SimpleNamespace(message=message)]


@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-tests")
    import agent
    agent._GRAPH = None  # force a fresh graph per test
    return agent


def test_agent_not_configured_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import agent
    agent._GRAPH = None
    result = agent.run_chat("hello", history=[])
    assert "GROQ_API_KEY" in result["answer"]
    assert result["tool_calls"] == []


def test_agent_single_tool_round_trip(agent_module):
    class FakeClient:
        def __init__(self):
            self.n = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, model, temperature, messages, tools, tool_choice):
            self.n += 1
            assert temperature == 0.1  # per project requirement: low, deterministic temp
            if self.n == 1:
                tc = _FakeToolCall("c1", "compare_sales", {"dimension": "category", "top_n": 5})
                return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            assert tool_msgs
            top = json.loads(tool_msgs[-1]["content"])["ranked"][0]["group"]
            return _FakeResponse(_FakeMessage(content=f"{top} had the highest sales.", tool_calls=None))

    client = FakeClient()
    agent_module._get_client = lambda: client
    result = agent_module.run_chat("Which category had the highest sales?", history=[])

    assert "highest sales" in result["answer"]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "compare_sales"


def test_agent_multi_step_tool_chain(agent_module):
    class FakeClient:
        def __init__(self):
            self.n = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, model, temperature, messages, tools, tool_choice):
            self.n += 1
            if self.n == 1:
                tc = _FakeToolCall("c1", "compare_sales", {"dimension": "category", "top_n": 1})
                return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))
            if self.n == 2:
                tc = _FakeToolCall("c2", "get_forecast_series", {})
                return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))
            return _FakeResponse(_FakeMessage(content="Done.", tool_calls=None))

    client = FakeClient()
    agent_module._get_client = lambda: client
    result = agent_module.run_chat("top category, then list forecastable series", history=[])

    assert [tc["name"] for tc in result["tool_calls"]] == ["compare_sales", "get_forecast_series"]


def test_agent_surfaces_tool_error_instead_of_fabricating(agent_module):
    class FakeClient:
        def __init__(self):
            self.n = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, model, temperature, messages, tools, tool_choice):
            self.n += 1
            if self.n == 1:
                tc = _FakeToolCall("c1", "forecast_predict", {
                    "target_date": "2025-01-01", "sku": "BOGUS", "region": "PL-Central", "channel": "Retail",
                    "price_unit": 5.0, "promotion_flag": False, "delivery_days": 3,
                    "stock_available": 150, "delivered_qty": 170,
                })
                return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            err = json.loads(tool_msgs[-1]["content"])
            assert "error" in err  # the tool correctly reported failure, not a fake number
            return _FakeResponse(_FakeMessage(content=f"I can't do that: {err['error']}", tool_calls=None))

    client = FakeClient()
    agent_module._get_client = lambda: client
    result = agent_module.run_chat("Forecast BOGUS for 2025-01-01", history=[])

    assert "predicted_units_sold" not in result["answer"]
    tool_result = json.loads(result["tool_calls"][0]["result"])
    assert "error" in tool_result


def test_agent_groq_failure_handled_gracefully(agent_module):
    class FailingClient:
        chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("outage")))
        )

    agent_module._get_client = lambda: FailingClient()
    result = agent_module.run_chat("hello", history=[])
    assert "error" in result["answer"].lower()
    assert result["tool_calls"] == []


def test_agent_forwards_conversation_history_for_followups(agent_module):
    class HistoryCheckClient:
        def __init__(self):
            self.seen_messages = None
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, model, temperature, messages, tools, tool_choice):
            self.seen_messages = messages
            return _FakeResponse(_FakeMessage(content="Milk forecast is X.", tool_calls=None))

    client = HistoryCheckClient()
    agent_module._get_client = lambda: client
    history = [
        {"role": "user", "content": "Which category had the highest sales in 2024?"},
        {"role": "assistant", "content": "Milk."},
    ]
    agent_module.run_chat("What about its forecast?", history=history)

    contents = [m.get("content", "") for m in client.seen_messages]
    assert any("Milk" in c for c in contents)
    assert any("its forecast" in c for c in contents)


def test_agent_max_rounds_prevents_infinite_tool_loop(agent_module):
    """If the model keeps requesting tools forever, the graph must stop instead
    of looping indefinitely."""
    class AlwaysToolClient:
        def __init__(self):
            self.n = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, model, temperature, messages, tools, tool_choice):
            self.n += 1
            tc = _FakeToolCall(f"c{self.n}", "get_metadata", {})
            return _FakeResponse(_FakeMessage(content="", tool_calls=[tc]))

    client = AlwaysToolClient()
    agent_module._get_client = lambda: client
    result = agent_module.run_chat("loop forever", history=[])
    assert len(result["tool_calls"]) == agent_module.MAX_TOOL_ROUNDS
