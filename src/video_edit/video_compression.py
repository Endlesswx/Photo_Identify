import os
import subprocess
import concurrent.futures
from pathlib import Path
import time
from tqdm import tqdm
import threading

# ================= 最终极速配置 =================
SOURCE_DIR = Path(r"F:\图片\iPhone相册") 
OUTPUT_DIR = Path(r"E:\Caches\相册视频-压缩后")


# 【火力全开】
# 既然之前 CPU 能跑 99%，我们直接给到 14 线程
# 配合 ultrafast 模式，彻底吃满每一个 CPU 周期
MAX_COMPUTE_WORKERS = 12
# ===============================================

def process_video(task_args):
    input_file, output_file = task_args
    
    # --- 1. 过滤逻辑 ---
    # 过滤同名 JPG (Live Photo)
    try:
        if input_file.with_suffix(".jpg").exists():
            return (0, None)
    except: pass

    # 过滤已完成文件 (大于10KB才算真正完成，防止之前的1KB尸体干扰)
    if output_file.exists():
        if output_file.stat().st_size > 10 * 1024: 
            return (0, None)
        else:
            try: output_file.unlink() # 删除之前的 1KB 坏文件
            except: pass

    try:
        # --- 2. 获取时长 ---
        cmd_probe = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(input_file)]
        res = subprocess.run(cmd_probe, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        duration = float(res.stdout.strip()) if res.stdout.strip() else 0
        if duration == 0: return (2, f"⚠️ 元数据读失败: {input_file.name}")

        # --- 3. 动态帧率策略 (修正版) ---
        if duration < 1.0:
            # 【修复】对于小于1秒的极短视频 (如Live Photo瞬间)，强制保留原始帧率或至少 5fps
            # 防止 fps=2 导致 0.1秒的视频采不到帧
            fps = "5" 
        elif duration <= 30: fps = "2"
        elif duration <= 120: fps = "1"
        elif duration <= 300: fps = "0.2"
        else: fps = "0.1"

        # --- 4. FFmpeg 命令构建 (修复核心) ---
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 【关键修复1】scale过滤器：trunc(oh*a/2)*2 
        # 这个公式的意思是：先算出宽度，除以2取整再乘以2，强行变成偶数，防止 "270" 这种奇数导致崩溃
        scale_filter = f"scale=trunc(oh*a/2)*2:480,fps={fps}"
        
        cmd = [
            "ffmpeg", "-y", "-i", str(input_file), 
            "-vf", scale_filter, 
            "-c:v", "libx264", 
            "-preset", "ultrafast", 
            "-crf", "32",
            "-pix_fmt", "yuv420p", # 【关键修复2】强制 8bit 兼容模式，解决 HDR P010 问题
            "-threads", "1", 
            "-an", "-loglevel", "error",
            str(output_file)
        ]

        subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        
        # --- 5. 产物校验 (防止 1KB 尸体) ---
        if not output_file.exists() or output_file.stat().st_size < 5 * 1024:
            # 如果生成的文件小于 5KB，肯定也是坏的，删掉报错
            if output_file.exists(): output_file.unlink()
            return (2, f"❌ 转换失败(文件过小): {input_file.name}")

        return (1, None)

    except Exception as e:
        if output_file.exists():
            try: output_file.unlink()
            except: pass
        return (2, f"❌ {input_file.name}: {str(e)}")

def main():
    if not SOURCE_DIR.exists(): return

    print("🔍 扫描文件...")
    all_files = list(SOURCE_DIR.rglob("*"))
    valid_files = [f for f in all_files if f.suffix.lower() in ('.mov', '.mp4')]
    
    print(f"🚀 最终极速版 (CPU 满载模式) | 视频数: {len(valid_files)}")
    print(f"⚡ 线程数: {MAX_COMPUTE_WORKERS} | 策略: 直读 + 过滤同名JPG")
    print("-" * 50)

    tasks = []
    for f in valid_files:
        try:
            out_f = OUTPUT_DIR / f.relative_to(SOURCE_DIR).with_suffix(".mp4")
            tasks.append((f, out_f))
        except: pass

    start_time = time.time()
    stats = {"processed": 0, "skipped": 0, "error": 0}

    # 使用 ThreadPoolExecutor 管理 FFmpeg 子进程
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_COMPUTE_WORKERS) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_video, task): task for task in tasks}
        
        pbar = tqdm(total=len(tasks), unit="vid", dynamic_ncols=True)
        
        for future in concurrent.futures.as_completed(future_to_file):
            status, msg = future.result()
            if status == 0: 
                stats["skipped"] += 1
            elif status == 1: 
                stats["processed"] += 1
            else: 
                stats["error"] += 1
                tqdm.write(msg)
            pbar.update(1)
        pbar.close()

    print(f"\n🎉 耗时: {time.time() - start_time:.2f}s | 成功: {stats['processed']} | 跳过: {stats['skipped']} | 错误: {stats['error']}")

if __name__ == "__main__":
    main()