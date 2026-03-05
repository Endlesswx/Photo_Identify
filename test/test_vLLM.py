import asyncio
import os
import sys
import time
import json
import aiohttp
import subprocess

# 1. 确保能加载项目内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from photo_identify.llm import async_call_image_model

# ================= 配置区 =================
IMAGE_DIR = r"C:\Users\wx\Desktop\test图片"
API_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "../models/qwen3.5-9b-awq"
VALID_EXTS = ('.jpg', '.png', '.jpeg', '.webp')

# 压测并发范围：1 到 12
CONCURRENCY_RANGE = range(1, 13) 
# ==========================================

def get_vram_usage():
    """实时抓取 NVIDIA 显存占用 (MiB)"""
    try:
        # 使用 nvidia-smi 查询当前显存使用量
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        return int(output.strip())
    except Exception:
        return 0

async def process_task(sem, session, filename, results):
    """单个图片推理任务"""
    async with sem:
        img_path = os.path.join(IMAGE_DIR, filename)
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        
        start = time.perf_counter()
        # 调用你项目中的多模态解析逻辑
        res = await async_call_image_model(
            image_bytes=img_bytes, 
            session=session,
            base_url=API_URL, 
            model=MODEL_NAME
        )
        duration = time.perf_counter() - start
        
        # 提取 Token 统计
        usage = res.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        results.append({"tokens": tokens, "time": duration})

async def run_step(concurrency, image_files):
    """运行特定并发等级的测试"""
    sem = asyncio.Semaphore(concurrency)
    results = []
    
    # 记录开始前的显存
    vram_start = get_vram_usage()
    start_time = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_task(sem, session, f, results) for f in image_files]
        await asyncio.gather(*tasks)
    
    total_duration = time.perf_counter() - start_time
    vram_end = get_vram_usage()
    
    total_tokens = sum(r['tokens'] for r in results)
    system_tps = total_tokens / total_duration if total_duration > 0 else 0
    
    return {
        "concurrency": concurrency,
        "duration": total_duration,
        "tps": system_tps,
        "vram": vram_end
    }

async def main():
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(VALID_EXTS)]
    if not files:
        print("❌ 错误：指定目录没有图片文件。")
        return

    print(f"🔥 开始阶梯压测 | 硬件: RTX 5060 Ti 16GB")
    print(f"📦 样本数量: {len(files)} 张 | 模型: {MODEL_NAME}")
    print("-" * 65)
    print(f"{'并发数':<8} | {'总耗时(s)':<10} | {'系统吞吐(tok/s)':<15} | {'实时显存(MiB)':<12}")
    print("-" * 65)

    test_logs = []
    for c in CONCURRENCY_RANGE:
        # 每一轮中间停顿一下，让显存回收稳定
        await asyncio.sleep(1)
        
        stats = await run_step(c, files)
        test_logs.append(stats)
        
        print(f"{stats['concurrency']:<9} | {stats['duration']:<11.2f} | {stats['tps']:<16.2f} | {stats['vram']:<12}")

    # --- 结果分析 ---
    print("\n" + "🏆 压测分析总结 " + "="*35)
    best_tps = max(test_logs, key=lambda x: x['tps'])
    print(f"🌟 最佳性能并发点: {best_tps['concurrency']}")
    print(f"🚀 峰值系统吞吐量: {best_tps['tps']:.2f} tokens/s")
    print(f"📈 显存峰值占用: {max(log['vram'] for log in test_logs)} MiB")
    print("=" * 51)

if __name__ == "__main__":
    asyncio.run(main())