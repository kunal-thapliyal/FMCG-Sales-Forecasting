"""
Phase 3 — LangGraph agent that turns natural-language questions into calls
against the existing FastAPI/analytics/forecasting stack (via chat_tools.py)
and a final natural-language answer.

Kept intentionally simple, per project instructions: one agent node, one
tool-execution node, a two-edge loop between them, done. No multi-agent
graph, no separate planner/critic, no vector store.

GROQ_API_KEY is read lazily (only when the graph actually runs), not at
import time — so importing this module (e.g. from main.py at app startup)
never fails just because the key isn't set. Only an actual /chat request
without a configured key returns a clear configuration error.
"""
import json
import operator
import os
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv

import chat_tools

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0.1"))
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the "AI FMCG Analyst" — a natural-language assistant over an \
FMCG (fast-moving consumer goods) sales analytics and forecasting system.

Rules you must follow:
1. You NEVER invent, estimate, or calculate business numbers (sales totals, revenue, \
inventory levels, promotion uplift, forecasts) yourself. Every number in your answer \
must come from a tool call result. If you haven't called a tool for a number, don't \
state it.
2. The historical sales dataset ends at the max_date returned by get_metadata (2024). \
There is no actual/recorded sales data beyond that date. For any date after max_date, \
use the forecast_predict tool and describe the result with words like "forecasted", \
"predicted", or "expected units sold" — never call it "actual sales" or imply it \
already happened.
3. Before calling forecast_predict, if the user hasn't given price/promotion/delivery/\
stock/delivered-qty values, call get_forecast_last_known first to get sensible current \
defaults for that (sku, region, channel), then use those in forecast_predict.
4. If a tool call returns an error (invalid SKU, insufficient history, unsupported \
dimension, etc.), explain the error plainly to the user in your final answer. Do not \
silently retry with made-up values and do not fabricate a result instead.
5. If the question genuinely cannot be answered with the available tools (e.g. it asks \
about something outside sales/inventory/promotions/forecasting), say so directly \
rather than guessing.
6. For follow-up questions ("what about its forecast?", "compare it to..."), use the \
conversation history to resolve what "it"/"that" refers to (e.g. a category or SKU \
named earlier in the conversation).
7. Keep answers concise and business-focused — lead with the number/finding, then a \
short supporting detail. Avoid restating the entire raw tool payload.
8. For optional tool parameters such as filters and dates, omit the parameter entirely \
when the user did not specify it. Do not send null for optional filters unless the tool \
schema explicitly requires it."""



class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    rounds: int


def _get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from groq import Groq
    return Groq(api_key=api_key)


def _agent_node(state: AgentState) -> dict:
    client = _get_client()
    if client is None:
        # Should not normally be reached — /chat checks this before invoking
        # the graph — but guard here too in case the graph is called directly.
        return {
            "messages": [{
                "role": "assistant",
                "content": "The AI Analyst isn't configured: GROQ_API_KEY is missing on the server.",
            }],
        }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            messages=messages,
            tools=chat_tools.TOOLS,
            tool_choice="auto",
        )
    # except Exception as e:
    #     return {
    #         "messages": [{
    #             "role": "assistant",
    #             "content": f"The AI Analyst hit an error calling the language model: {e}",
    #         }],
    #     }
    except Exception as e:
        print(f"[AI ANALYST] Groq error: {type(e).__name__}: {e}")
        return {
            "messages": [{
                "role": "assistant",
                "content": "The AI Analyst encountered an error while processing the request.",
                }],
            }


    choice = response.choices[0].message
    msg = {"role": "assistant", "content": choice.content or ""}
    if choice.tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in choice.tool_calls
        ]
    return {"messages": [msg], "rounds": state.get("rounds", 0) + 1}


def _tools_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    results = []
    for tc in last.get("tool_calls", []):
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        content = chat_tools.call_function(name, args)
        results.append({"role": "tool", "tool_call_id": tc["id"], "name": name, "content": content})
    return {"messages": results}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if last.get("role") == "assistant" and last.get("tool_calls") and state.get("rounds", 0) < MAX_TOOL_ROUNDS:
        return "tools"
    return "end"


def _build_graph():
    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


def is_configured() -> bool:
    return bool(os.environ.get("GROQ_API_KEY"))


def run_chat(message: str, history: List[dict]) -> dict:
    """Runs one turn of the AI FMCG Analyst.

    `history` is a list of {"role": "user"|"assistant", "content": str} from
    prior turns (session-level memory only, passed in by the caller — no
    server-side session store). Returns
    {"answer": str, "tool_calls": [{"name": ..., "arguments": ..., "result": ...}]}.
    """
    if not is_configured():
        return {
            "answer": (
                "The AI Analyst isn't configured yet: the server is missing GROQ_API_KEY. "
                "The rest of the dashboard is unaffected — ask your administrator to set "
                "GROQ_API_KEY and restart the backend to enable the chat."
            ),
            "tool_calls": [],
        }

    # Keep only the most recent conversation turns.
# This preserves follow-up context without allowing the chat history
# to grow indefinitely.
        
    clean_history = []
    for m in history:
        role = m.get("role")
        content = m.get("content")

        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean_history.append({
                "role": role,
                "content": content.strip(),
            })

# Keep the latest 6 messages (3 user/assistant exchanges).
    clean_history = clean_history[-6:]

    initial_messages = clean_history + [
        {
            "role": "user",
            "content": message.strip(),
        }
    ]

    # clean_history = [
    #     {"role": m["role"], "content": m.get("content", "")}
    #     for m in history if m.get("role") in ("user", "assistant") and m.get("content")
    # ]
    # initial_messages = clean_history + [{"role": "user", "content": message}]

    try:
        final_state = _graph().invoke({"messages": initial_messages, "rounds": 0})
    except Exception as e:
        return {"answer": f"The AI Analyst hit an unexpected error: {e}", "tool_calls": []}

    tool_calls_log = []
    answer = ""
    msgs = final_state["messages"]
    tool_results_by_id = {m["tool_call_id"]: m for m in msgs if m.get("role") == "tool"}
    for m in msgs:
        if m.get("role") == "assistant":
            if m.get("content"):
                answer = m["content"]
            for tc in m.get("tool_calls", []):
                result_msg = tool_results_by_id.get(tc["id"])
                tool_calls_log.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "result": result_msg["content"] if result_msg else None,
                })

    if not answer:
        answer = "I wasn't able to produce a response for that question — please try rephrasing it."

    return {"answer": answer, "tool_calls": tool_calls_log}
