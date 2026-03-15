"""构建便携源码分发包。

产物: dist/photo_identify_portable.zip
内部根目录: photo_identify_portable/
  src/             — 源码
  pyproject.toml
  uv.lock
  install.bat
  start_gui.bat
  uv/uv.exe        — 内置 uv
  bin/             — 外部工具（如有）
"""
from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"
PACKAGE_DIR = DIST_DIR / "photo_identify_portable"
ZIP_PATH = DIST_DIR / "photo_identify_portable"  # shutil.make_archive 自动加 .zip

FILES = [
    "pyproject.toml",
    "uv.lock",
    "install.bat",
    "start_gui.bat",
]

DIRS = [
    "src",
]


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", "*.egg-info",
    ))


def bundle_uv(dst_dir: Path) -> None:
    """将 uv.exe 复制到分发包的 uv/ 目录。"""
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("[警告] 未在 PATH 中找到 uv，跳过内置 uv")
        return
    uv_dir = dst_dir / "uv"
    uv_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(uv_path, uv_dir / "uv.exe")
    print("已内置 uv:", uv_path)


def main() -> None:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    for name in FILES:
        src = PROJECT_ROOT / name
        if src.exists():
            shutil.copy2(src, PACKAGE_DIR / name)

    for name in DIRS:
        copy_tree(PROJECT_ROOT / name, PACKAGE_DIR / name)

    # 内置 uv
    bundle_uv(PACKAGE_DIR)

    # bin/ 目录（外部工具，可选）
    bin_dir = PROJECT_ROOT / "bin"
    if bin_dir.exists():
        copy_tree(bin_dir, PACKAGE_DIR / "bin")

    # 打包为 zip
    zip_file = shutil.make_archive(
        str(ZIP_PATH),     # 输出路径（不含 .zip）
        "zip",
        root_dir=DIST_DIR, # zip 内以 photo_identify_portable/ 为根
        base_dir="photo_identify_portable",
    )

    # 清理临时目录
    shutil.rmtree(PACKAGE_DIR)

    print("便携分发包已生成:", zip_file)


if __name__ == "__main__":
    main()
