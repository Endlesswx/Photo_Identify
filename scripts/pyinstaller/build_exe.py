from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
TOOLS_DIR = DIST_DIR / "photo_identify" / "tools"
BIN_DIR = PROJECT_ROOT / "bin"
MAIN_EXE = DIST_DIR / "photo_identify" / "photo_identify.exe"
ROOT_MAIN_EXE = DIST_DIR / "photo_identify.exe"

SPECS = [
    "photo_identify.spec",
    "video_compression.spec",
    "lvip_decompression.spec",
]


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    for spec in SPECS:
        run(["uv", "run", "--with", "pyinstaller", "pyinstaller", spec, "--noconfirm", "--clean"])

    # 布局：dist/photo_identify/{photo_identify.exe,_internal,tools,bin,data}
    tools_dir = TOOLS_DIR
    tools_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(BIN_DIR, DIST_DIR / "photo_identify" / "bin")

    # 清理旧 helper
    for helper in ("video_compression.exe", "lvip_decompression.exe"):
        tool_path = tools_dir / helper
        if tool_path.exists():
            tool_path.unlink(missing_ok=True)

    # 复制 helper exe 到 tools 目录
    for helper in ("video_compression.exe", "lvip_decompression.exe"):
        helper_path = DIST_DIR / helper
        if helper_path.exists():
            copy_file(helper_path, tools_dir / helper)
            helper_path.unlink(missing_ok=True)

    # 删除根目录的 photo_identify.exe，仅保留 onedir 入口
    if MAIN_EXE.exists():
        ROOT_MAIN_EXE.unlink(missing_ok=True)

    print("Portable package prepared at:", DIST_DIR / "photo_identify")


if __name__ == "__main__":
    main()
