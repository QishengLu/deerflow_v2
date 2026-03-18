# CLAUDE.md — deer-flow-v2 (DeerFlow Agent)

LangGraph v2 ReAct agent，通过 `create_agent()` 构建单一线性 ReAct 推理链。

---

## 架构

- **框架**: LangGraph v2 + LangChain ReAct
- **模型**: ChatAnthropic（claude-sonnet-4-6 via shubiaobiao API）
- **图拓扑**: 单一 ReAct agent（无复杂 StateGraph 节点）
- **中间件**（RCA eval 模式仅用 2 个）:
  - `DanglingToolCallMiddleware` — 补丁缺失的 ToolMessage
  - `SummarizationMiddleware` — 15,564 tokens 触发上下文摘要（保留最近 10 条消息）

完整应用（非 eval）还有 ThreadData、Uploads、Sandbox、TodoList、Title、Memory、ViewImage、Subagent、Clarification 等中间件。

---

## 入口文件

| 文件 | 用途 |
|------|------|
| `backend/agent_runner.py` | RolloutRunner 统一接口（stdin/stdout） |
| `src/agents/` | Agent 创建逻辑 |
| `config.yaml` | 模型、工具、沙箱、中间件配置 |
| `backend/.env` | API Key / Base URL |

---

## agent_runner.py 工作流程

1. 接收 stdin JSON（question, system_prompt, user_prompt, data_dir）
2. **Prompt 注入**:
   - 合并 DeerFlow base prompt + RCA expert instructions
   - Regex 移除 think_tool 引用（框架不支持，"four→three tools"）
   - 追加 data_dir 提示到 system_prompt 和 user_prompt
3. 创建 ReAct agent（ChatAnthropic + 3 个 parquet tools）
4. 调用 agent，收集 trajectory
5. 输出 stdout JSON（output, trajectory, usage）

---

## 工具

RCA eval 模式使用 3 个 parquet 工具（直接 import，非 MCP）：

| 工具 | 来源 | 用途 |
|------|------|------|
| `list_tables_in_directory` | Deep_Research/src/rca_tools.py | 列出 parquet 文件 |
| `get_schema` | 同上 | 获取表 schema |
| `query_parquet_files` | 同上 | SQL 查询 parquet |

完整应用还有 `ls`, `read_file`, `write_file`, `str_replace`, `bash`, `task`（子 agent）等沙箱工具。

---

## 关键配置（config.yaml）

```yaml
model: claude-sonnet-4-6
openai_api_base: https://api.shubiaobiao.cn/v1
sandbox: LocalSandboxProvider
summarization: enabled (15564 tokens)
memory: disabled      # eval 模式避免跨样本污染
title_generation: disabled
```

---

## 特殊处理

| 项目 | 说明 |
|------|------|
| think_tool | ❌ 不支持，agent_runner.py 中 regex 过滤 |
| UsageTracker | install_anthropic_hooks()（ChatAnthropic 直接走 Anthropic SDK） |
| 上下文压缩 | SummarizationMiddleware，15564 tokens 触发 |
| 数据目录 | 追加 `CRITICAL DATA LOCATION` 到 prompt，确保 agent 不乱找文件 |

---

## 环境

```bash
cd /home/nn/SOTA-agents/deer-flow-v2
uv run python backend/agent_runner.py  # stdin/stdout 接口
```

- Python 3.12+，uv 管理
- 依赖: langchain, langchain-anthropic, langgraph, fastapi, pydantic

---

## 已知问题

- **不稳定（~33% 成功率，kimi-k2 冒烟测试）**: agent 有时无法找到 data_dir，盲目搜索文件系统
- **切换 sonnet-4.6 后待验证**: 模型能力提升可能改善 data_dir 定位问题

---

## RolloutRunner 配置

```yaml
# RolloutRunner/configs/agents/deerflow.yaml
name: deerflow
cmd: ["uv", "run", "python", "backend/agent_runner.py"]
cwd: /home/nn/SOTA-agents/deer-flow-v2
exp_id: deerflow-claude-sonnet-4.6
agent_type: deerflow
concurrency: 5
timeout: 600
```
