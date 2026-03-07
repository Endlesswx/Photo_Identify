import asyncio
import os
import sys
import time
import json
import base64
import aiohttp
import subprocess

# 1. 确保能加载项目内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ================= 配置区 =================
VIDEO_DIR = r"C:\Users\wx\Desktop\Test视频转码后"
API_URL = r"http://127.0.0.1:8000/v1"
MODEL_NAME = "../models/qwen3.5-9b-awq" # 已根据 /v1/models 接口返回自动修正
VALID_EXTS = ('.mp4', '.mov')

# 压测并发范围：1 到 12
CONCURRENCY_RANGE = range(2, 7)
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

async def async_call_video_model(video_bytes: bytes, session: aiohttp.ClientSession, base_url: str, model: str) -> dict:
    """
    专门针对视频模型的 API 构造逻辑
    注意：此处的 video_url base64 格式是为了兼容类似于 Qwen2-VL 等通过 OpenAI 格式包装的图片/视频分析模型
    """
    video_b64 = base64.b64encode(video_bytes).decode('utf-8')
    # 构建请求 payload
    payload = {
        "model": model,
        "temperature": 0.0, # 压测时通常为了稳妥使用低温度
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请用中文详细描述这个视频的内容、场景以及主要发生的事件。"},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{video_b64}"
                        }
                    }
                ]
            }
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    start_req = time.perf_counter()
    try:
        async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as response:
            status = response.status
            try:
                data = await response.json()
            except aiohttp.ContentTypeError:
                raw = await response.text()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}
                    
            if status >= 400:
                print(f"  [API Error] HTTP {status}: {data}")
                return {"error": f"HTTP {status}"}
            
            return data
    except Exception as exc:
        print(f"  [Network Error] {str(exc)}")
        return {"error": str(exc)}

async def process_task(sem, session, filename, results):
    """单个视频推理任务"""
    async with sem:
        vid_path = os.path.join(VIDEO_DIR, filename)
        with open(vid_path, "rb") as f:
            vid_bytes = f.read()
        
        start = time.perf_counter()
        
        # 调用视频专用的接口发送请求
        res = await async_call_video_model(
            video_bytes=vid_bytes, 
            session=session,
            base_url=API_URL, 
            model=MODEL_NAME
        )
        duration = time.perf_counter() - start
        
        # 提取 Token 统计
        usage = res.get("usage", {})
        tokens = usage.get("completion_tokens", 0)
        results.append({"tokens": tokens, "time": duration})
        print(f"  [已完成] {filename} | 处理时间: {duration:.2f}s | 生成 Tokens: {tokens}")

async def run_step(concurrency, video_files):
    """运行特定并发等级的测试"""
    sem = asyncio.Semaphore(concurrency)
    results = []
    
    # 记录开始前的显存
    vram_start = get_vram_usage()
    start_time = time.perf_counter()
    
    # 因为视频文件较大，所以把限制连接的 connector 放大一些
    connector = aiohttp.TCPConnector(limit=50)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_task(sem, session, f, results) for f in video_files]
        await asyncio.gather(*tasks)
    
    total_duration = time.perf_counter() - start_time
    vram_end = get_vram_usage()
    
    total_tokens = sum(r.get('tokens', 0) for r in results)
    system_tps = total_tokens / total_duration if total_duration > 0 else 0
    
    return {
        "concurrency": concurrency,
        "duration": total_duration,
        "tps": system_tps,
        "vram": vram_end
    }

async def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 错误：指定目录不存在 => {VIDEO_DIR}")
        return

    files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(VALID_EXTS)]
    if not files:
        print(f"❌ 错误：指定目录 '{VIDEO_DIR}' 内没有找到有效的视频文件。")
        return

    print(f"🔥 开始视频模型阶梯压测 | 硬件: RTX 设备")
    print(f"📦 样本数量: {len(files)} 个视频 | 模型: {MODEL_NAME}")
    global_start_time = time.time()
    print("-" * 65)
    print(f"{'并发数':<8} | {'总耗时(s)':<10} | {'系统吞吐(tok/s)':<15} | {'实时显存(MiB)':<12}")
    print("-" * 65)

    test_logs = []
    for c in CONCURRENCY_RANGE:
        # 每一轮中间停顿一下，让显存回收稳定
        await asyncio.sleep(2)
        
        # 如果文件数不足以满足并发，其实多余的并发也没有意义
        actual_files = files[:c * 2] if len(files) > c * 2 else files
        
        stats = await run_step(c, actual_files)
        test_logs.append(stats)
        
        print(f"{stats['concurrency']:<9} | {stats['duration']:<11.2f} | {stats['tps']:<16.2f} | {stats['vram']:<12}")

    # --- 结果分析 ---
    if test_logs:
        print("\n" + "🏆 压测分析总结 " + "="*35)
        best_tps = max(test_logs, key=lambda x: x['tps'])
        print(f"🌟 最佳性能并发点: {best_tps['concurrency']}")
        print(f"🚀 峰值系统吞吐量: {best_tps['tps']:.2f} tokens/s")
        print(f"📈 显存峰值占用: {max(log['vram'] for log in test_logs)} MiB")
        print(f"⏱️ 测试总耗时: {time.time() - global_start_time:.2f} s")
        print("=" * 51)
    else:
        print("无有效压测结果。")

if __name__ == "__main__":
    asyncio.run(main())
