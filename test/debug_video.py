import os
import subprocess
from pathlib import Path
import shutil

# ================= 配置 =================
SOURCE_DIR = Path(r"F:\图片\iPhone相册")
# 将要诊断的文件名（根据你报错的日志填写的）
TARGET_FILENAME = "IMG_3094_20250105110053.mov"
# =======================================

def analyze_file():
    print(f"🔍 正在全盘搜索 {TARGET_FILENAME} ...")
    found_files = list(SOURCE_DIR.rglob(TARGET_FILENAME))
    
    if not found_files:
        print("❌ 未找到该文件，请确认文件名是否正确。")
        return

    input_file = found_files[0]
    print(f"✅ 找到文件: {input_file}")
    print(f"📂 文件大小: {input_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 1. 运行 ffprobe 查看文件内部结构
    print("\n" + "="*20 + " [1. 文件流信息 (ffprobe)] " + "="*20)
    cmd_probe = [
        "ffprobe", 
        "-v", "warning", 
        "-show_streams", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        str(input_file)
    ]
    subprocess.run(cmd_probe)

    # 2. 模拟真实转码 (开启详细日志)
    print("\n" + "="*20 + " [2. 转码尝试 (FFmpeg)] " + "="*20)
    output_test = Path("debug_output.mp4")
    
    # 使用之前提供的“终极修复版”参数
    cmd_ffmpeg = [
        "ffmpeg", "-y", 
        "-i", str(input_file),
        "-vf", "scale=trunc(oh*a/2)*2:480,fps=1", # 模拟 1fps
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
        "-pix_fmt", "yuv420p", # 强制 8bit
        "-an", # 去除音频
        str(output_test)
    ]
    
    # 这里的关键是 stderr=None，让错误信息直接打印到屏幕上
    result = subprocess.run(cmd_ffmpeg, text=True)
    
    if output_test.exists():
        size_kb = output_test.stat().st_size / 1024
        print(f"\n📊 输出文件大小: {size_kb:.2f} KB")
        if size_kb < 5:
            print("❌ 依然失败：文件过小 (这是问题的关键)")
        else:
            print("✅ 居然成功了？(可能是多线程并发导致的问题)")
        
        # 清理垃圾
        output_test.unlink()
    else:
        print("❌ 输出文件未生成")

if __name__ == "__main__":
    analyze_file()