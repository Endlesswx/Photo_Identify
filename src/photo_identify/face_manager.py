"""人脸识别模块，封装 insightface 进行特征提取与聚类。"""

import logging
import threading
from typing import Any, Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# 使用锁确保单例实例化
_instance_lock = threading.Lock()
_app_instance = None
_device_mode = "初始化中..."

def get_device_mode() -> str:
    """获取当前硬件加速状态（仅在 get_face_app 调用后准确）。"""
    return _device_mode

def get_face_app():
    """懒加载获取 InsightFace 的 FaceAnalysis 实例，并动态处理 providers。"""
    global _app_instance, _device_mode
    if _app_instance is not None:
        return _app_instance

    with _instance_lock:
        if _app_instance is not None:
            return _app_instance

        import insightface
        import onnxruntime as ort

        try:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                # 尝试加载 CUDA 驱动
                try:
                    app = insightface.app.FaceAnalysis(
                        name='buffalo_l', 
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    app.prepare(ctx_id=0, det_size=(640, 640))
                    
                    # 验证是否真的使用了 CUDA
                    is_cuda_active = False
                    if app.models:
                        # 检查第一个模型的 session 实际加载的 provider
                        first_model = list(app.models.values())[0]
                        if hasattr(first_model, 'session') and 'CUDAExecutionProvider' in first_model.session.get_providers():
                            is_cuda_active = True
                            
                    if is_cuda_active:
                        _app_instance = app
                        _device_mode = "GPU 加速 (CUDA)"
                        logger.info("InsightFace 成功使用 CUDA 加速")
                        return _app_instance
                    else:
                        logger.warning(
                            "==============================================================\n"
                            "⚠️ CUDA 加载失败，系统已自动回退至 CPU 模式。\n"
                            "常见原因：系统缺少 CUDA 12.x 或 cuDNN 9.x，或相关 DLL（如 cublasLt64_12.dll）未加入环境变量 PATH。\n"
                            "👉 注意：由于底层 C++ 引擎的报错无法直接输出到此界面的日志框中，\n"
                            "请您去【启动本程序的黑色终端控制台窗口】中查看红字报错（[ONNXRuntimeError] : 1 : FAIL : Error loading...），\n"
                            "那里会明确告诉您具体缺失了哪个 dll 文件！\n"
                            "=============================================================="
                        )
                        # 销毁之前创建的 app
                        del app
                except Exception as e:
                    logger.warning(f"加载 CUDAExecutionProvider 发生异常，回退到 CPU: {e}")
            
            # 无论是因为没有装 GPU 版，还是 GPU 加载报错，统一回退到 CPU
            # 限制 CPU 线程数，避免满载导致系统卡顿
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # 保留至少 2 个逻辑核心给系统，最多使用总核心数的一半
            allowed_threads = max(1, min(cpu_count // 2, cpu_count - 2))
            
            # 配置 session options 限制内部线程
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = allowed_threads
            sess_options.inter_op_num_threads = allowed_threads
            
            import insightface.model_zoo.model_zoo
            # 必须自定义 provider_options 或直接修改内部实例化，但 InsightFace 的 python API 
            # 默认封装了 SessionOptions。为了注入，我们可以使用 kwargs 传递
            
            # 注意：InsightFace App 初始化目前较难直接透传 sess_options，
            # 最有效的方法是设置全局环境变量控制 onnxruntime 的 OpenMP 行为，
            # 或者通过修改环境参数。
            import os
            # 设置 OpenMP/MKL 等底层计算库的线程数，必须在 import numpy / onnxruntime 之前或尽早设置
            # 因为这里可能是懒加载，某些库可能已经被 import 过了。
            os.environ["OMP_NUM_THREADS"] = str(allowed_threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(allowed_threads)
            os.environ["MKL_NUM_THREADS"] = str(allowed_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(allowed_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(allowed_threads)
            
            # InsightFace 内部使用 kwargs 传递给 InferenceSession
            kwargs = {
                "sess_options": sess_options
            }
            
            app = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider']
            )
            # 通过强制修改模型实例内部的 session options 重新加载模型
            try:
                for model_name, model in app.models.items():
                    if hasattr(model, 'model_path'):
                        # 强制使用我们限制过线程的 session 覆盖默认的
                        model.session = ort.InferenceSession(
                            model.model_path, 
                            sess_options=sess_options, 
                            providers=['CPUExecutionProvider']
                        )
                        model.get_inputs()
            except Exception as opt_err:
                logger.debug(f"调整 CPU session 线程数失败 (忽略): {opt_err}")

            app.prepare(ctx_id=-1, det_size=(640, 640))
            _app_instance = app
            _device_mode = f"CPU 模式 (已限流至 {allowed_threads} 线程)"
            logger.info("InsightFace 运行在 CPU 模式")
            return _app_instance
            
        except Exception as e:
            logger.error(f"InsightFace 初始化失败: {e}")
            _device_mode = "不可用"
            return None


def extract_faces(image_np: np.ndarray) -> List[Dict[str, Any]]:
    """提取图片中的人脸。

    Args:
        image_np: 图片的 Numpy 数组 (RGB格式, BGR由内部转换处理或要求传入BGR)
                  注意：InsightFace 默认输入是 BGR 格式的图像。

    Returns:
        包含每个人脸信息的列表：
        [
            {
                "bbox": [x1, y1, x2, y2],
                "embedding": np.ndarray (512,),
                "det_score": float
            },
            ...
        ]
    """
    app = get_face_app()
    if app is None:
        return []

    # insightface 要求输入 BGR
    # 如果外部传入的是 PIL 的 RGB image_np，需转换为 BGR
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        bgr_img = image_np[:, :, ::-1].copy()
    else:
        bgr_img = image_np

    try:
        faces = app.get(bgr_img)
        results = []
        for face in faces:
            results.append({
                "bbox": face.bbox.tolist(),
                "embedding": face.embedding,
                "det_score": float(face.det_score)
            })
        return results
    except Exception as e:
        logger.error(f"提取人脸失败: {e}")
        return []


def cluster_face_embeddings(embeddings_with_ids: List[Tuple[int, np.ndarray]], eps: float = 1.0, min_samples: int = 5) -> Dict[int, int]:
    """使用 DBSCAN 对人脸进行聚类。
    
    InsightFace 的 buffalo_l (ArcFace) 输出的 embedding 通常是 L2 归一化过的 512 维向量。
    可以使用余弦距离或欧氏距离。这里基于 L2 归一化后的欧氏距离。

    Args:
        embeddings_with_ids: 列表元素为 (face_id, embedding_array)
        eps: DBSCAN 的距离阈值。1.0 对应的余弦相似度约为 0.5，能非常强力地将同一人物各种不同角度、表情、光线下的照片合并，极大地减少被分散成多个的问题。
        min_samples: 形成聚类的最少样本数。设置为 5，意味着如果一张脸在相册里出现次数极少（比如路人、极度模糊导致偏离重心的废脸），将被视为噪音丢弃，不单独为您生成一个无用的人物分组。

    Returns:
        字典，映射 face_id 到 cluster_id。
    """
    if not embeddings_with_ids:
        return {}

    from sklearn.cluster import DBSCAN
    
    ids = [item[0] for item in embeddings_with_ids]
    X = np.array([item[1] for item in embeddings_with_ids])
    
    # 对特征向量进行 L2 归一化（ArcFace默认应该是归一化的，以防万一再做一次）
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-10)

    # 运行 DBSCAN
    # metric='euclidean' 在 L2 归一化的数据上等价于基于余弦相似度
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    labels = clusterer.fit_predict(X)

    # labels 为 -1 表示噪点（未分类）
    return {face_id: int(label) for face_id, label in zip(ids, labels)}
