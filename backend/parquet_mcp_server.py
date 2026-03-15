"""
parquet_mcp_server.py — RCA Parquet Tools MCP Server (stdio transport)

Exposes three tools via MCP protocol:
  - list_tables_in_directory : discover parquet files recursively
  - get_schema               : column schema for one or multiple files
  - query_parquet_files      : DuckDB SQL query against parquet files

Logic ported 1:1 from Deep_Research/src/rca_tools.py.
Run via: uv run python parquet_mcp_server.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Union

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("parquet-rca")

TOKEN_LIMIT = 5000

ALLOWED_STEMS = {
    "normal_logs",
    "abnormal_logs",
    "normal_traces",
    "abnormal_traces",
    "normal_metrics",
    "abnormal_metrics",
    "normal_metrics_histogram",
    "abnormal_metrics_histogram",
    "normal_metrics_sum",
    "abnormal_metrics_sum",
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _serialize_datetime(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(i) for i in obj]
    return obj


def _estimate_token_count(text: str) -> int:
    return (len(text) + 2) // 3


def _enforce_token_limit(payload: str, context: str) -> str:
    token_estimate = _estimate_token_count(payload)
    if token_estimate <= TOKEN_LIMIT:
        return payload
    current_size = len(json.loads(payload)) if payload.startswith("[") else None
    suggested_limit = None
    if current_size:
        ratio = TOKEN_LIMIT / token_estimate
        suggested_limit = max(1, int(current_size * ratio * 0.8))
    warning = {
        "error": "Result exceeds token budget",
        "context": context,
        "estimated_tokens": token_estimate,
        "token_limit": TOKEN_LIMIT,
        "rows_returned": current_size,
        "suggested_limit": suggested_limit,
        "suggestion": (
            "Reduce LIMIT value"
            + (f" (try LIMIT {suggested_limit})" if suggested_limit else "")
            + ", filter with WHERE, select fewer columns, or use aggregation."
        ),
    }
    return json.dumps(warning, ensure_ascii=False, indent=2)


def _sanitize_column_name(name: str) -> str:
    return name.replace(".", "_")


def _build_rename_select(parquet_path: str) -> str:
    """Return SELECT clause that renames dot-containing columns."""
    import duckdb

    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_path}') LIMIT 0")
        columns = [desc[0] for desc in result.description]
    finally:
        conn.close()

    if not any("." in col for col in columns):
        return "*"

    parts = []
    for col in columns:
        if "." in col:
            parts.append(f'"{col}" AS {_sanitize_column_name(col)}')
        else:
            parts.append(col)
    return ", ".join(parts)


def _get_schema_one(parquet_file: str) -> dict:
    import duckdb

    if not Path(parquet_file).exists():
        return {"error": f"File not found: {parquet_file}"}
    conn = duckdb.connect(":memory:")
    try:
        result = conn.execute(f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 0")
        schema = [{"name": _sanitize_column_name(d[0]), "type": str(d[1])} for d in result.description]
        row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()[0]
        return {"file": parquet_file, "row_count": row_count, "columns": schema}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


# ── MCP tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
def list_tables_in_directory(directory: str) -> str:
    """List all parquet files in a directory (recursively) with metadata.

    Args:
        directory: Directory path to search for parquet files.

    Returns:
        JSON array of {filename, path, row_count, column_count}.
    """
    import duckdb

    dir_path = Path(directory)
    if not dir_path.exists():
        return json.dumps({"error": f"Directory not found: {directory}"})
    if not dir_path.is_dir():
        return json.dumps({"error": f"Not a directory: {directory}"})

    files_info = []
    for file_path in sorted(dir_path.rglob("*.parquet")):
        if file_path.stem not in ALLOWED_STEMS:
            continue
        try:
            conn = duckdb.connect(":memory:")
            row_count = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()[0]
            result = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
            column_count = len(result.description)
            conn.close()
            files_info.append(
                {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "row_count": row_count,
                    "column_count": column_count,
                }
            )
        except Exception as e:
            files_info.append({"filename": file_path.name, "path": str(file_path), "error": str(e)})

    return _enforce_token_limit(json.dumps(files_info, ensure_ascii=False, indent=2), "list_tables_in_directory")


@mcp.tool()
def get_schema(parquet_files: Union[str, List[str]]) -> str:
    """Get column schema for one or more parquet files.

    Args:
        parquet_files: Single path string, or list of path strings.

    Returns:
        JSON with {file, row_count, columns:[{name,type}]}, or array for multiple files.
    """
    if isinstance(parquet_files, str):
        return _enforce_token_limit(json.dumps(_get_schema_one(parquet_files), ensure_ascii=False, indent=2), "get_schema")
    return _enforce_token_limit(
        json.dumps([_get_schema_one(f) for f in parquet_files], ensure_ascii=False, indent=2),
        "get_schema",
    )


@mcp.tool()
def query_parquet_files(parquet_files: Union[str, List[str]], query: str, limit: int = 10) -> str:
    """Query parquet files with DuckDB SQL. Each file is registered as a view named after its stem.
    Column names containing dots are renamed (e.g. attr.http.status → attr_http_status).

    Args:
        parquet_files: Path(s) to parquet file(s).
        query: SQL query (use file stem as table name, e.g. SELECT * FROM abnormal_metrics LIMIT 5).
        limit: Max rows returned (default 10).

    Returns:
        JSON array of row dicts, or error object with suggestions.
    """
    import duckdb

    if isinstance(parquet_files, str):
        parquet_files = [parquet_files]

    for fp in parquet_files:
        if not Path(fp).exists():
            return json.dumps({"error": f"File not found: {fp}. Use list_tables_in_directory first."})

    conn = duckdb.connect(":memory:")
    table_names: set = set()
    try:
        for file_path in parquet_files:
            base = Path(file_path).stem
            name = base
            counter = 1
            while name in table_names:
                name = f"{base}_{counter}"
                counter += 1
            table_names.add(name)
            select_clause = _build_rename_select(file_path)
            conn.execute(f"CREATE VIEW {name} AS SELECT {select_clause} FROM read_parquet('{file_path}')")

        rows_raw = conn.execute(query).fetchall()
        columns = [d[0] for d in conn.description]
        rows = [dict(zip(columns, r)) for r in rows_raw]
        rows = _serialize_datetime(rows)
        if len(rows) > limit:
            rows = rows[:limit]
        return _enforce_token_limit(json.dumps(rows, ensure_ascii=False, indent=2), "query_parquet_files")
    except Exception as e:
        return json.dumps(
            {
                "error": str(e),
                "query": query,
                "available_tables": list(table_names),
            }
        )
    finally:
        conn.close()


if __name__ == "__main__":
    mcp.run()
