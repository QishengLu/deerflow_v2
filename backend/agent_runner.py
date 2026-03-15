#!/usr/bin/env python
"""
agent_runner.py — DeerFlow v2 RCA Rollout Interface

stdin:  JSON {question, system_prompt, user_prompt,
              compress_system_prompt, compress_user_prompt, data_dir}
stdout: JSON {output (CausalGraph JSON), trajectory (OpenAI format)}
exit 0 on success, non-zero on failure.

Prompt injection:
  - system_prompt (RCA_ANALYSIS_SP) → appended to v2's system prompt
  - user_prompt   (RCA_ANALYSIS_UP, already contains incident_description) → HumanMessage
  - compress_*    → not used (no compress node in v2)
  - question      → already embedded in user_prompt via {incident_description}
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/nn/SOTA-agents/RolloutRunner")
from src.usage_tracker import UsageTracker

_tracker = UsageTracker()
_tracker.install_openai_hooks()
_tracker.install_anthropic_hooks()  # ChatAnthropic 走 Anthropic SDK，需要单独 hook

# 清理 RolloutRunner 路径和 src 模块缓存，避免与 deerflow 的 src 包冲突
sys.path.remove("/home/nn/SOTA-agents/RolloutRunner")
for mod_name in list(sys.modules):
    if mod_name == "src" or mod_name.startswith("src."):
        del sys.modules[mod_name]

# Add backend/ to sys.path so `from src.xxx import ...` works (same as debug.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

# Load .env: check backend/ first, then deer-flow-v2/ root
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

# Set config path BEFORE any deerflow imports that trigger lazy config loading
os.environ.setdefault(
    "DEER_FLOW_CONFIG_PATH",
    str(Path(__file__).resolve().parent.parent / "config.yaml"),
)

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# 直接导入 parquet 工具（LangChain @tool 格式），不走 MCP
# 复用 Deep_Research 的 rca_tools.py（已验证稳定）
sys.path.insert(0, "/home/nn/SOTA-agents/Deep_Research/src")
from rca_tools import list_tables_in_directory, get_schema, query_parquet_files
sys.path.remove("/home/nn/SOTA-agents/Deep_Research/src")


# ── model factory ────────────────────────────────────────────────────────────


def _make_summarization_middleware() -> SummarizationMiddleware | None:
    """Create SummarizationMiddleware using ChatAnthropic to avoid Bedrock format issues.

    Replicates _create_summarization_middleware() from deerflow's agent.py but
    uses ChatAnthropic instead of the config-defined ChatOpenAI.
    """
    from src.config.summarization_config import get_summarization_config

    config = get_summarization_config()
    if not config.enabled:
        return None

    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [t.to_tuple() for t in config.trigger]
        else:
            trigger = config.trigger.to_tuple()

    keep = config.keep.to_tuple()
    kwargs = {
        "model": _make_anthropic_model(),
        "trigger": trigger,
        "keep": keep,
    }
    if config.trim_tokens_to_summarize is not None:
        kwargs["trim_tokens_to_summarize"] = config.trim_tokens_to_summarize
    if config.summary_prompt is not None:
        kwargs["summary_prompt"] = config.summary_prompt
    return SummarizationMiddleware(**kwargs)


def _make_anthropic_model(max_tokens: int = 32768) -> ChatAnthropic:
    """Create ChatAnthropic pointing to shubiaobiao API (Anthropic-compatible endpoint)."""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.shubiaobiao.cn/v1")
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=base_url,
        max_tokens=max_tokens,
    )


# ── helpers ──────────────────────────────────────────────────────────────────


def _extract_text(content) -> str:
    """Extract plain text from AIMessage content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def strip_markdown_json(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers from LLM output."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def to_openai_message(msg) -> dict | None:
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": _extract_text(msg.content)}

    if isinstance(msg, AIMessage):
        tool_calls = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["args"], ensure_ascii=False),
                },
            }
            for tc in (msg.tool_calls or [])
        ]
        entry: dict = {"role": "assistant", "content": _extract_text(msg.content)}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    if isinstance(msg, ToolMessage):
        return {"role": "tool", "content": str(msg.content), "tool_call_id": msg.tool_call_id}

    return None


def convert_trajectory(messages: list) -> list[dict]:
    return [m for msg in messages if (m := to_openai_message(msg)) is not None]


# ── main ─────────────────────────────────────────────────────────────────────


async def run(payload: dict) -> dict:
    system_prompt = payload["system_prompt"]  # RCA_ANALYSIS_SP (formatted with date)
    user_prompt = payload["user_prompt"]      # RCA_ANALYSIS_UP (formatted with incident_description)
    data_dir = payload.get("data_dir", "")

    # 过滤 think_tool 相关内容（本框架无 think_tool，避免 LLM 尝试调用不存在的工具）
    system_prompt = re.sub(r"  4\. \*\*think_tool\*\*.*\n", "", system_prompt)
    system_prompt = system_prompt.replace("four tools", "three tools")

    # Append data_dir to BOTH system_prompt (survives summarization) and user_prompt
    if data_dir:
        data_hint = (
            f"\n\nCRITICAL DATA LOCATION: The telemetry data for this incident is stored in: `{data_dir}`\n"
            f"You MUST use this exact path. Start by calling `list_tables_in_directory` with directory=\"{data_dir}\"."
        )
        system_prompt += data_hint
        user_prompt += data_hint

    # 直接使用导入的 parquet 工具（不走 MCP，避免 MCP server 启动失败导致空工具列表）
    tools = [list_tables_in_directory, get_schema, query_parquet_files]

    # Build v2 system prompt and append RCA expert instructions
    from src.agents.lead_agent.prompt import apply_prompt_template
    v2_system_prompt = apply_prompt_template(subagent_enabled=False)
    combined_system_prompt = v2_system_prompt + "\n\n---\n\n" + system_prompt

    # Build runtime config
    config = {
        "configurable": {
            "thread_id": f"rca-eval-{id(payload)}",
            "model_name": "claude-sonnet",
            "thinking_enabled": False,
            "is_plan_mode": False,
            "subagent_enabled": False,
        },
        "recursion_limit": 100,
    }

    # Build agent components — only keep RCA-relevant middleware
    from src.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware

    model = _make_anthropic_model()

    # Minimal middleware: DanglingToolCall (safety) + Summarization (token management)
    # Use _make_summarization_middleware() so summarization uses ChatAnthropic, not ChatOpenAI.
    # (ChatOpenAI → shubiaobiao → Bedrock fails with content-format 400 when summarization triggers)
    middleware = [DanglingToolCallMiddleware()]
    summarization_mw = _make_summarization_middleware()
    if summarization_mw is not None:
        middleware.append(summarization_mw)

    agent = create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        system_prompt=combined_system_prompt,
    )

    # Invoke: HumanMessage = user_prompt (already contains incident_description)
    state = {"messages": [HumanMessage(content=user_prompt)]}

    result = await agent.ainvoke(state, config=config)

    messages = result.get("messages", [])
    trajectory = convert_trajectory(messages)

    # Final output: last AIMessage without tool_calls
    final_content = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_content = _extract_text(msg.content)
            break

    result = {
        "output": strip_markdown_json(final_content),
        "trajectory": trajectory,
        "usage": _tracker.get_usage(),
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return result


def main():
    try:
        payload = json.loads(sys.stdin.read())
        asyncio.run(run(payload))
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
