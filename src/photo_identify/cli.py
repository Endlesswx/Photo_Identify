"""命令行入口：提供 scan / search / stats / export / import-json 子命令。

使用 argparse 组织子命令，是用户与本工具交互的唯一入口：
  uv run python -m photo_identify scan --path <目录>
  uv run python -m photo_identify search "关键词"
  uv run python -m photo_identify stats
  uv run python -m photo_identify export --output backup.json
  uv run python -m photo_identify import-json --input photo_analysis.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from photo_identify.config import (
    DEFAULT_BASE_URL,
    DEFAULT_DB_PATH,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEXT_MODEL,
    DEFAULT_RPM_LIMIT,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TPM_LIMIT,
    DEFAULT_WORKERS,
    load_api_key,
)


def _setup_logging(verbose: bool):
    """配置日志输出级别。

    Args:
        verbose: True 时输出 DEBUG 级别日志，否则 WARNING。
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cmd_scan(args):
    """执行 scan 子命令：扫描目录并分析图片。

    Args:
        args: argparse 解析后的参数对象。
    """
    from photo_identify.scanner import scan

    api_key = load_api_key(args.api_key)
    if not api_key:
        print("错误: 缺少 API Key，请设置环境变量 SILICONFLOW_API_KEY 或使用 --api-key", file=sys.stderr)
        sys.exit(1)

    scan(
        paths=args.path,
        db_path=str(args.db),
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        rpm_limit=args.rpm,
        tpm_limit=args.tpm,
        workers=args.workers,
    )


def _cmd_search(args):
    """执行 search 子命令：搜索已分析的图片。

    Args:
        args: argparse 解析后的参数对象。
    """
    from photo_identify.search import format_results, search

    api_key = load_api_key(args.api_key)
    results, warnings = search(
        query=args.query,
        db_paths=str(args.db),
        limit=args.limit,
        smart=args.smart,
        api_key=api_key,
        base_url=args.base_url,
        local_expand=True,
    )
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}", file=sys.stderr)
    print(format_results(results))


def _cmd_stats(args):
    """执行 stats 子命令：显示数据库统计信息。

    Args:
        args: argparse 解析后的参数对象。
    """
    from photo_identify.storage import Storage

    db_file = Path(args.db)
    if not db_file.exists():
        print(f"数据库文件不存在: {db_file}")
        return

    storage = Storage(args.db)
    count = storage.count()
    size_kb = db_file.stat().st_size / 1024
    storage.close()
    print(f"数据库: {db_file}")
    print(f"文件大小: {size_kb:.1f} KB")
    print(f"图片记录数: {count}")


def _cmd_export(args):
    """执行 export 子命令：将 SQLite 数据导出为 JSON 文件。

    Args:
        args: argparse 解析后的参数对象。
    """
    from photo_identify.storage import Storage

    storage = Storage(args.db)
    records = storage.all_records()
    storage.close()

    output_path = Path(args.output)
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已导出 {len(records)} 条记录到 {output_path}")


def _cmd_import_json(args):
    """执行 import-json 子命令：从旧版 JSON 文件导入数据到 SQLite。

    Args:
        args: argparse 解析后的参数对象。
    """
    from photo_identify.image_utils import compute_md5, read_image_bytes
    from photo_identify.storage import Storage

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print("JSON 格式错误: 预期为数组", file=sys.stderr)
        sys.exit(1)

    storage = Storage(args.db)
    imported = 0
    skipped = 0

    for item in data:
        md5 = item.get("md5", "")
        if not md5:
            # 旧数据没有 md5，尝试从文件计算
            path = item.get("path", "")
            if path and Path(path).exists():
                try:
                    md5 = compute_md5(read_image_bytes(path))
                except Exception:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

        if storage.has_md5(md5):
            skipped += 1
            continue

        # 解析旧格式的 LLM 结果
        llm = item.get("llm", {})
        parsed = llm.get("parsed", {})
        metadata = item.get("metadata", {})

        record = {
            "path": item.get("path", ""),
            "file_name": item.get("file_name", ""),
            "size_bytes": item.get("size_bytes"),
            "md5": md5,
            "sha256": item.get("sha256", ""),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "image_mode": metadata.get("mode", ""),
            "image_format": metadata.get("format", ""),
            "exif": metadata.get("exif", {}),
            "created_time": item.get("created_time", ""),
            "modified_time": item.get("modified_time", ""),
            "scene": parsed.get("scene", ""),
            "objects": parsed.get("objects", []),
            "style": parsed.get("style", ""),
            "location_time": parsed.get("location_time", ""),
            "wallpaper_hint": parsed.get("wallpaper_hint", ""),
            "llm_raw": llm.get("raw", ""),
            "analyzed_at": datetime.now().isoformat(),
        }
        storage.upsert(record)
        imported += 1

    storage.close()
    print(f"导入完成 — 新增: {imported}  跳过: {skipped}")


def main():
    """解析命令行参数并分发到对应子命令。"""
    parser = argparse.ArgumentParser(
        prog="photo_identify",
        description="图片扫描分析与自然语言检索工具",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="数据库文件路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── gui ───────────────────────────────────────────
    p_gui = subparsers.add_parser("gui", help="启动图形交互界面")
    
    def _cmd_gui(args):
        from photo_identify.gui import launch_gui
        # db path is handled globally by --db but we pass the arg.db directly
        launch_gui(args.db)
        
    p_gui.set_defaults(func=_cmd_gui)

    p_scan = subparsers.add_parser("scan", help="扫描目录并分析图片")
    p_scan.add_argument("--path", nargs="+", required=True, help="要扫描的目录（支持多个）")
    p_scan.add_argument("--api-key", default="", help="API Key")
    p_scan.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API Base URL")
    p_scan.add_argument("--model", default=DEFAULT_IMAGE_MODEL, help="模型名称")
    p_scan.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p_scan.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_scan.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p_scan.add_argument("--rpm", type=int, default=DEFAULT_RPM_LIMIT, help="每分钟最大请求数")
    p_scan.add_argument("--tpm", type=int, default=DEFAULT_TPM_LIMIT, help="每分钟最大 token 数")
    p_scan.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并发线程数")
    p_scan.set_defaults(func=_cmd_scan)

    # ── search ─────────────────────────────────────────
    p_search = subparsers.add_parser("search", help="搜索已分析的图片")
    p_search.add_argument("query", help="搜索文本")
    p_search.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="最大返回条数")
    p_search.add_argument("--smart", action="store_true", help="启用 LLM 辅助关键词扩展")
    p_search.add_argument("--api-key", default="", help="API Key（smart 模式需要）")
    p_search.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p_search.set_defaults(func=_cmd_search)

    # ── stats ──────────────────────────────────────────
    p_stats = subparsers.add_parser("stats", help="显示数据库统计信息")
    p_stats.set_defaults(func=_cmd_stats)

    # ── export ─────────────────────────────────────────
    p_export = subparsers.add_parser("export", help="导出为 JSON 文件")
    p_export.add_argument("--output", default="photo_export.json", help="输出文件路径")
    p_export.set_defaults(func=_cmd_export)

    # ── import-json ────────────────────────────────────
    p_import = subparsers.add_parser("import-json", help="从旧版 JSON 导入数据")
    p_import.add_argument("--input", required=True, help="旧版 JSON 文件路径")
    p_import.set_defaults(func=_cmd_import_json)

    args = parser.parse_args()
    _setup_logging(args.verbose)
    args.func(args)


if __name__ == "__main__":
    main()
