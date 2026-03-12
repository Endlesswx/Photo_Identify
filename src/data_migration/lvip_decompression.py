import os
import zipfile
import io
import time
import shutil

# ================= 配置区域 =================
# 你的机械硬盘照片目录路径 (注意 Windows 路径前的 r)
SOURCE_DIR = r"F:\图片\iPhone相册"

# 批处理大小 (个文件)
# 针对机械硬盘优化：
# 设置为 50-100 可以让磁头一次性顺序读取约 1GB-2GB 数据进入内存
# 然后一次性顺序写入，避免磁头疯狂跳动。
# 你的内存有 48GB，这个数值非常安全。
# 200时内存占用约12G，处理1825个文件355秒
BATCH_SIZE = 200
# ===========================================


def get_all_livp_files(directory):
    """预先扫描所有文件，避免边处理边扫描导致磁头抖动"""
    print("正在扫描目录建立文件索引 (HDD优化)...")
    livp_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".livp"):
                full_path = os.path.join(root, file)
                # 过滤掉已经是 0KB 的文件
                if os.path.getsize(full_path) > 0:
                    livp_list.append(full_path)
    print(f"扫描完成，共发现 {len(livp_list)} 个待处理文件。")
    return livp_list


def process_batch(batch_files):
    """
    核心优化逻辑：
    1. 批量读入 RAM (连续读)
    2. 内存解压 (CPU计算)
    3. 批量写入 Disk (连续写)
    """
    # 存储读取到的二进制数据: [(path, bytes, stat), ...]
    read_buffer = []
    # 存储待写入的操作: [('write', path, bytes), ('mkdir', path), ('zero', path, stat)]
    write_ops = []

    # --- 阶段 1: 批量读取 (顺序读) ---
    # print(f"  > 正在读取 {len(batch_files)} 个文件到内存...")
    for file_path in batch_files:
        try:
            # 获取文件状态（主要是时间戳），用于稍后恢复
            stat = os.stat(file_path)
            with open(file_path, "rb") as f:
                content = f.read()
            read_buffer.append((file_path, content, stat))
        except Exception as e:
            print(f"[读取失败] {file_path}: {e}")

    # --- 阶段 2: 内存处理 (CPU 计算，不占 IO) ---
    # print(f"  > 正在内存中解压处理...")
    for file_path, content, stat in read_buffer:
        try:
            zip_obj = io.BytesIO(content)
            if not zipfile.is_zipfile(zip_obj):
                print(f"[跳过] 不是有效的 ZIP: {file_path}")
                continue

            with zipfile.ZipFile(zip_obj, "r") as z:
                # 寻找图片和视频
                file_list = z.namelist()
                img_file = next(
                    (
                        f
                        for f in file_list
                        if f.lower().endswith((".jpg", ".jpeg", ".heic"))
                    ),
                    None,
                )
                mov_file = next(
                    (f for f in file_list if f.lower().endswith(".mov")), None
                )

                if img_file:
                    # 路径计算
                    root_dir = os.path.dirname(file_path)
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]

                    # 目标文件夹: .../Photos/2023-02-18.../
                    target_folder = os.path.join(root_dir, base_name)

                    # 准备创建文件夹操作
                    write_ops.append(("mkdir", target_folder, None))

                    # 统一文件名前缀
                    # 逻辑：取 livp 内部图片的主文件名，如 IMG_4269
                    internal_base = os.path.basename(img_file).split(".")[0]
                    if not internal_base:
                        internal_base = "image"  # 防止空名

                    # 确定图片后缀 (修复 .HEIC.JPG 问题)
                    img_ext = os.path.splitext(img_file)[1].lower()
                    if ".heic" in img_file.lower() and ".jpg" in img_file.lower():
                        img_ext = ".jpg"

                    # 准备写入图片
                    target_img_path = os.path.join(
                        target_folder, f"{internal_base}{img_ext}"
                    )
                    img_data = z.read(img_file)
                    write_ops.append(("write", target_img_path, img_data))

                    # 准备写入视频 (如果有)
                    if mov_file:
                        target_mov_path = os.path.join(
                            target_folder, f"{internal_base}.mov"
                        )
                        mov_data = z.read(mov_file)
                        write_ops.append(("write", target_mov_path, mov_data))

                    # 准备替换原文件为 0KB (保留时间戳)
                    write_ops.append(("zero", file_path, stat))

        except Exception as e:
            print(f"[解压错误] {file_path}: {e}")

    # --- 阶段 3: 批量写入 (顺序写) ---
    # print(f"  > 正在写入磁盘...")
    processed_count = 0
    for op_type, path, data in write_ops:
        try:
            if op_type == "mkdir":
                if not os.path.exists(path):
                    os.makedirs(path)

            elif op_type == "write":
                # 如果文件已存在，跳过写入（防止重复IO）
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(data)

            elif op_type == "zero":
                # 删除原文件
                if os.path.exists(path):
                    os.remove(path)
                # 创建 0KB 空文件
                with open(path, "w") as f:
                    pass
                # 恢复时间戳 (mtime)
                # data 在这里存储的是之前的 os.stat 对象
                if data:
                    os.utime(path, (data.st_atime, data.st_mtime))
                processed_count += 1

        except Exception as e:
            print(f"[写入错误] {path}: {e}")

    return processed_count


def main():
    print("==========================================")
    print("   LIVP 极速原地解压工具 (HDD 优化版)    ")
    print("==========================================")

    start_global = time.time()

    # 1. 获取任务列表
    all_files = get_all_livp_files(SOURCE_DIR)
    total_files = len(all_files)

    if total_files == 0:
        print("没有发现需要处理的文件。")
        return

    processed_total = 0

    # 2. 分批处理
    for i in range(0, total_files, BATCH_SIZE):
        batch = all_files[i : i + BATCH_SIZE]

        # 简单进度条
        print(f"[{i+1}/{total_files}] 正在处理批次 ({len(batch)} 个文件)...")

        batch_start = time.time()
        count = process_batch(batch)
        batch_end = time.time()

        processed_total += count

        # 打印本批次速度
        speed = len(batch) / (batch_end - batch_start + 0.001)
        print(
            f"   完成。批次耗时: {batch_end - batch_start:.1f}s (速度: {speed:.1f} 个/秒)"
        )

    end_global = time.time()
    duration = end_global - start_global

    print("\n==========================================")
    print(f"全部任务完成！")
    print(f"共处理文件: {processed_total}")
    print(f"总耗时: {duration:.2f} 秒")
    print("==========================================")


if __name__ == "__main__":
    # 再次确认路径是否存在
    if os.path.exists(SOURCE_DIR):
        main()
    else:
        print(f"错误: 找不到路径 {SOURCE_DIR}")
