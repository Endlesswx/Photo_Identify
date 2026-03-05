# 遍历指定文件夹，识别视频格式，并按照要求的区间（30秒内、1分钟内、每分钟递增）进行归类统计

import os
import cv2
from collections import defaultdict

# 视频文件夹路径
target_folder = r'F:\图片\iPhone相册'

def get_video_duration(file_path):
    """获取视频时长（秒）"""
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps > 0:
            return frame_count / fps
    except Exception as e:
        print(f"无法读取文件 {file_path}: {e}")
    finally:
        cap.release()
    return 0

def analyze_videos(folder_path):
    # 支持的视频后缀
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm')
    
    format_counts = defaultdict(int)
    duration_counts = defaultdict(int) # key 为区间描述
    total_count = 0

    if not os.path.exists(folder_path):
        print("❌ 文件夹路径不存在！")
        return

    print(f"🔍 正在扫描: {folder_path} ...\n")

    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                file_path = os.path.join(root, file)
                duration = get_video_duration(file_path)
                
                # 统计格式
                format_counts[ext] += 1
                total_count += 1

                # 统计时长区间
                if duration <= 30:
                    duration_counts["0-30秒"] += 1
                elif duration <= 60:
                    duration_counts["31-60秒"] += 1
                else:
                    # 计算是第几分钟 (例如: 61-120秒归为2分钟)
                    minutes = int((duration - 0.1) // 60) + 1
                    duration_counts[f"{minutes}分钟"] += 1

    # --- 打印结果 ---
    print("📊 --- 统计报告 ---")
    print(f"✅ 总视频个数: {total_count}")
    
    print("\n📂 格式分布:")
    for ext, count in sorted(format_counts.items()):
        print(f"  - {ext}: {count} 个")

    print("\n⏳ 时长分布:")
    # 对时长区间进行简单排序输出
    sorted_durations = sorted(duration_counts.items(), key=lambda x: (len(x[0]), x[0]))
    for label, count in sorted_durations:
        print(f"  - {label}: {count} 个")

if __name__ == "__main__":
    # 填入你的文件夹路径
    analyze_videos(target_folder)