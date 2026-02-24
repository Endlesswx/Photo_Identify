# Photo_Identify 工程化优化 — 完成总结

## 第一轮：架构重构

将单文件 `app.py`（554行）重构为 8 个模块的工程化架构。

```
src/photo_identify/
├── cli.py           # CLI 入口（scan/search/stats/export/import-json）
├── scanner.py       # 并发扫描主流程
├── storage.py       # SQLite + FTS5 存储层
├── llm.py           # LLM API 调用 + 重试 + 速率控制
├── image_utils.py   # 图片压缩/元数据/哈希
├── search.py        # 自然语言搜索
├── config.py        # 全局配置
└── __main__.py      # python -m 入口
```

| 问题 | 旧方案 | 新方案 |
|------|--------|--------|
| 存储 | 单 JSON 全量序列化 | SQLite 增量写入 |
| 搜索 | 无 | FTS5 + LIKE 双层搜索 |
| 并发 | 串行 | ThreadPoolExecutor |
| 去重 | 按路径 | 按 MD5 内容哈希 |
| 上传 | 原始大图 | 压缩至 ≤1280px JPEG |
| 重试 | 无 | 3 次指数退避 |

---

## 第二轮：智能搜索与深度优化

### 1. 智能搜索修复

**问题**：`--smart "美女在唱歌"` 返回空结果

**根因**：
- CLI 层未自动读取环境变量中的 API Key，导致 LLM 扩展被静默跳过
- LLM 提示词只做"关键词提取"，无法跨越语义鸿沟（"美女" vs "女性"，"唱歌" vs "录音室"）

**修复**：
- CLI 始终读取环境变量，缺少 key 时给出明确提示
- 重写 LLM 提示词为**同义词/近义词扩展**，例如 "美女在唱歌" → "美女 女性 女孩 唱歌 歌手 演唱 录音 麦克风 音乐"
- 修复后命中 5 张相关图片 ✅

### 2. 数据库表结构说明

| 表名 | 类型 | 说明 |
|------|------|------|
| `images` | 普通表 | 主表，存储图片元数据 + LLM 分析结果（20 个字段） |
| `images_fts` | FTS5 虚拟表 | 全文搜索索引，映射 images 表的 6 个文本字段 |
| `images_fts_data` | FTS5 影子表 | 存储分词后的倒排索引数据（BLOB 格式） |
| `images_fts_idx` | FTS5 影子表 | B-Tree 索引，加速 FTS 查找 |
| `images_fts_docsize` | FTS5 影子表 | 每条记录的文档大小（用于排名计算） |
| `images_fts_config` | FTS5 影子表 | FTS5 配置信息（版本号等） |

> [!NOTE]
> `images_fts_data` 的 `block` 字段显示为"乱码"是**正常行为**。它是 BLOB（二进制大对象）类型，SQLite FTS5 内部使用二进制格式存储倒排索引数据，不是文本，用数据库查看工具以文本模式查看自然显示为乱码。这些影子表由 FTS5 引擎自动管理，不需要也不应该手动操作。

### 3. 其他优化

| 修复项 | 说明 |
|--------|------|
| `RateLimiter` 线程安全 | 添加 `threading.Lock`，防止多线程并发调用时出现竞态条件 |
| `Ctrl+C` 中断处理 | `KeyboardInterrupt` 捕获后优雅退出，已分析的数据不丢失 |

### 4. 万级海量图片生产优化 (NEW)

为保障针对 10000+ 张图片实际应用的极致稳定性，已做以下排查与防御性加固：

- **架构级防锁机制**：虽然是并发分析，但我们巧妙地利用了多消费者单生产者的架构，所有的 SQLite 写入严格限制在主线程。彻底消除了 SQLite 高并发情况下的竞态与锁死问题。
- **SQLite 超时控制**：增加了 `timeout=30.0`，即使在极端长耗时写入的边缘情况下也能保证抗压能力。
- **API 强抗抖动**：将指数退避最大重试次数从 `3` 提升到 `5`，结合长连接确保几万次调用偶尔遇到 API 服务端拥堵（429 / 502 等）时不丢失进度。
- **控制台 I/O 防卡死**：为命令行进度条增加了 0.1s 的独立刷新节流控制，即使在数秒内以高速跳过成千上万张缓存图片时，也不会因为极其频繁的 print 拖慢主线程或卡死 Windows PowerShell 控制台。

---

## 使用方法

```bash
# 扫描图片目录
uv run python -m photo_identify scan --path "E:\图片" "D:\照片"
uv run python -m photo_identify scan --path "F:\图片"

# 普通搜索
uv run python -m photo_identify search "录音棚"

# 智能搜索（口语化查询，自动同义词扩展）
uv run python -m photo_identify search --smart "美女在唱歌"
uv run python -m photo_identify search --smart "手表"
uv run python -m photo_identify search --smart "小孩骑工程车"

# 查看统计
uv run python -m photo_identify stats

# 导入旧数据 / 导出
uv run python -m photo_identify import-json --input photo_analysis.json
uv run python -m photo_identify export --output backup.json
```
