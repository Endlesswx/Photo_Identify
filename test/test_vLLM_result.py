import asyncio
import os
import sys
import json
import base64
import re
import aiohttp

# 1. 确保能加载项目内部模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# ================= 配置区 =================
VIDEO_DIR = r"C:\Users\wx\Desktop\Test视频转码后"
API_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "../models/qwen3.5-9b-awq" 
# ==========================================

def extract_clean_content(data: dict) -> str:
    """
    清洗 vLLM 返回的 JSON 结果：
    1. 排除独立的 reasoning_content 字段
    2. 过滤 content 中的 <thought>...</thought> 或 <think>...</think> 标签
    """
    # 场景 A: 存在独立的推理字段 (如 DeepSeek 官方 API 或某些 vLLM 配置)
    # 我们只取最终回答内容
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    
    # 场景 B: 思考过程混在 content 中 (常见于 R1 系列模型)
    # 使用正则表达式剔除 <thought> 或 <think> 标签及其内部内容
    clean_content = re.sub(r'<(thought|think)>.*?</\1>', '', content, flags=re.DOTALL)
    
    # 去除首尾多余空白
    return clean_content.strip()

async def get_pure_result(video_path: str):
    """获取单个视频的纯净分析结果"""
    if not os.path.exists(video_path):
        print(f"❌ 文件不存在: {video_path}")
        return

    async with aiohttp.ClientSession() as session:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        
        video_b64 = base64.b64encode(video_bytes).decode('utf-8')
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请直接给出视频的分析结果，不要包含思考过程。"},
                        {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                        }
                    ]
                }
            ]
        }

        try:
            async with session.post(f"{API_URL}/chat/completions", json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    clean_res = extract_clean_content(data)
                    print(f"\n🎬 视频: {os.path.basename(video_path)}")
                    print(f"📝 结果:\n{clean_res}")
                    print("-" * 50)
                else:
                    print(f"❌ API 错误: {response.status}")
        except Exception as e:
            print(f"❌ 请求失败: {e}")

async def main():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 目录不存在: {VIDEO_DIR}")
        return

    # 获取目录下第一个视频进行演示
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov'))]
    if not videos:
        print("❌ 未发现视频文件")
        return

    print(f"🔍 正在从 {VIDEO_DIR} 获取纯净结果...")
    # 这里演示获取前 2 个视频的结果
    for vid in videos[:2]:
        await get_pure_result(os.path.join(VIDEO_DIR, vid))

if __name__ == "__main__":
    asyncio.run(main())
