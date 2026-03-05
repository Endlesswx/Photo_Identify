# 压缩视频分辨率及码率，供大模型分析

import os
import subprocess
import concurrent.futures
from pathlib import Path
import time
from tqdm import tqdm
import threading

# ================= 配置区域 =================
# 源目录 (机械硬盘)
SOURCE_DIR = Path(r"F:\图片\iPhone相册") 
# 输出目录 (SSD)
OUTPUT_DIR = Path(r"E:\Caches\相册视频-压缩后")

# 并行计算线程数 (Ultra 5 245KF 建议 12-14)
MAX_COMPUTE_WORKERS = 12

# 【关键优化】硬盘读取并发锁
# 机械硬盘最忌讳多线程随机读。限制为 2，强制让磁头尽量顺序读取。
# 48G 内存足够我们把视频读到内存里再处理。
DISK_READ_SEMAPHORE = threading.BoundedSemaphore(2) 
# ===========================================

def get_duration(file_path):
    """获取视频时长 (只读头部信息，速度快)"""
    try:
        # ffprobe 只读 header，对 IO 压力较小，但仍需加锁防止争抢
        with DISK_READ_SEMAPHORE:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return float(res.stdout.strip())
    except:
        return 0

def process_video(task_args):
    input_file, output_file = task_args
    
    # 1. 检查跳过逻辑 (同上)
    jpg_check = input_file.with_suffix(".jpg")
    if jpg_check.exists():
        return (0, f"⏭️  跳过 (JPG存在): {input_file.name}")
    if output_file.exists():
        return (0, f"⏭️  跳过 (已完成): {input_file.name}")

    # 2. 获取时长
    duration = get_duration(input_file)
    if duration == 0:
        return (2, f"⚠️  错误: 无法读取 {input_file.name}")

    # 3. 动态 FPS 策略
    if duration <= 30:
        fps = "2"
    elif duration <= 120:
        fps = "1"
    elif duration <= 300:
        fps = "0.2"
    else:
        fps = "0.1"

    # 4. 【核心优化】读取文件到内存
    # 如果文件小于 2GB，直接读入内存；否则走普通流式（防止爆内存，虽然你有48G）
    file_size_mb = input_file.stat().st_size / (1024 * 1024)
    video_data = None
    use_pipe = False

    try:
        if file_size_mb < 2048: # 2GB以下的文件全部进内存
            with DISK_READ_SEMAPHORE: # 加锁读取，保护 HDD
                with open(input_file, 'rb') as f:
                    video_data = f.read()
            use_pipe = True
        
        # 5. FFmpeg 命令构建
        # -i pipe:0  从标准输入读取数据
        # -preset ultrafast  极速模式 (比 veryfast 快 30-50%)
        # -threads 1  限制单进程线程数，减少上下文切换，依靠多进程并发
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        input_args = ["-i", "pipe:0"] if use_pipe else ["-i", str(input_file)]
        
        cmd = [
            "ffmpeg", "-y"] + input_args + [
            "-vf", f"scale=-2:480,fps={fps}", 
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
            "-threads", "1", # 让 CPU 核心专注于多任务并行
            "-an", "-loglevel", "error",
            str(output_file)
        ]

        # 执行命令
        # 如果使用 pipe，将 video_data 传入 input
        subprocess.run(
            cmd, 
            input=video_data if use_pipe else None,
            check=True, 
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        
        # 释放内存
        del video_data 
        return (1, f"✅ [{fps}fps] {input_file.name}")

    except Exception as e:
        return (2, f"❌ 失败: {input_file.name} - {str(e)}")

def main():
    if not SOURCE_DIR.exists():
        print(f"❌ 错误: 源目录不存在")
        return

    # 扫描文件
    all_files = list(SOURCE_DIR.rglob("*"))
    valid_files = [f for f in all_files if f.suffix.lower() in ('.mov', '.mp4')]
    
    print(f"🚀 极速模式启动 | 视频数: {len(valid_files)}")
    print(f"⚡ 优化策略: 内存缓冲(RAM Buffer) + 顺序读盘(Sequential Read) + UltraFast")
    print(f"💾 内存: 48GB (小文件全量载入) | CPU: 12 并发")
    print("-" * 50)

    tasks = []
    for f in valid_files:
        try:
            rel_path = f.relative_to(SOURCE_DIR)
            out_f = OUTPUT_DIR / rel_path.with_suffix(".mp4")
            tasks.append((f, out_f))
        except ValueError: pass

    start_time = time.time()
    stats = {"processed": 0, "skipped": 0, "error": 0}

    # 使用 ThreadPoolExecutor
    # 虽然 Python 有 GIL，但 subprocess 调用 FFmpeg 是独立的 OS 进程，不受 GIL 限制，且 I/O 也是释放 GIL 的
    # 这里用线程池管理 FFmpeg 子进程是最轻量高效的
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_COMPUTE_WORKERS) as executor:
        future_to_file = {executor.submit(process_video, task): task for task in tasks}
        pbar = tqdm(total=len(tasks), unit="vid", dynamic_ncols=True)

        for future in concurrent.futures.as_completed(future_to_file):
            status, msg = future.result()
            if status == 0: stats["skipped"] += 1
            elif status == 1: 
                stats["processed"] += 1
                # tqdm.write(msg) # 想看详细日志可以解开注释
            else: 
                stats["error"] += 1
                tqdm.write(msg)
            pbar.update(1)
        pbar.close()

    print(f"\n🎉 耗时: {time.time() - start_time:.2f}s | 成功: {stats['processed']} | 跳过: {stats['skipped']}")

if __name__ == "__main__":
    main()