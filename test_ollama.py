"""测试 Ollama 本地模型的文本对话能力，模拟重排序场景。

用法：
    uv run python test_ollama.py

会依次测试 models.db 中所有本地模型，发送与重排序完全相同的 prompt，
打印完整的 HTTP 状态码、响应体和解析结果，帮助定位问题。
"""

import json
import re
import urllib.request
import urllib.error
from photo_identify.model_manager import ModelManager, get_model_db_path

# 重排序系统提示词（与 search.py 完全一致）
RERANK_SYSTEM_PROMPT = """\
你是一个图片检索重排序助手。用户提供了一个搜索描述，以及若干候选图片的序号和内容信息。
你的任务是：深入理解用户的语言意图，从候选图片中选出相对最符合要求的图片，并按匹配程度从高到低排序返回。

返回规则严格遵守：
直接输出选中的候选图片序号的 JSON 数组，例如：
[3, 1, 5]
如果没有符合要求的，返回 []。不要输出任何除了方括号和数字之外的内容（不要加 ```json 标记）。"""

# 模拟的候选图片列表
MOCK_CANDIDATES = """[1] 场景: 一只橘猫趴在窗台上 | 包含物体: 猫, 窗台, 窗帘
[2] 场景: 一个小女孩在吃香蕉 | 包含物体: 女孩, 香蕉, 餐桌
[3] 场景: 公园里的花丛 | 包含物体: 花, 草地, 树木
[4] 场景: 海边日落全景 | 包含物体: 海, 太阳, 沙滩
[5] 场景: 小朋友在吃水果 | 包含物体: 小朋友, 水果, 草莓"""

MOCK_QUERY = "然然吃香蕉"
USER_CONTENT = f"真实搜索意图：{MOCK_QUERY}\n\n候选图片列表：\n{MOCK_CANDIDATES}"


def test_model(name, model_id, base_url, api_key=""):
    """测试单个模型的文本对话能力。"""
    print(f"\n{'='*60}")
    print(f"📌 测试模型: {name}")
    print(f"   model_id: {model_id}")
    print(f"   base_url: {base_url}")
    print(f"   api_key:  {'(空/本地)' if not api_key else api_key[:8] + '...'}")
    print(f"{'='*60}")

    payload = {
        "model": model_id,
        "temperature": 0.1,
        "max_tokens": 10240,
        "messages": [
            {"role": "system", "content": RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": USER_CONTENT},
        ],
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{base_url}/chat/completions"
    print(f"\n🔗 请求 URL: {url}")
    print(f"📦 请求 payload (model 字段): {payload['model']}")

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            raw_bytes = response.read()
            raw_text = raw_bytes.decode("utf-8")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw_bytes = exc.read()
        raw_text = raw_bytes.decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"\n❌ 网络异常: {type(exc).__name__}: {exc}")
        return

    print(f"\n📡 HTTP 状态码: {status}")
    print(f"📄 原始响应 ({len(raw_text)} 字符):")
    # 截断过长的响应
    if len(raw_text) > 2000:
        print(raw_text[:2000] + "\n... (截断)")
    else:
        print(raw_text)

    # 尝试解析
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"\n❌ 响应不是有效 JSON: {e}")
        return

    choices = data.get("choices", [])
    if not choices:
        print(f"\n❌ choices 为空! 完整响应: {json.dumps(data, ensure_ascii=False, indent=2)}")
        return

    msg = choices[0].get("message", {})
    content = msg.get("content", "")
    reasoning = msg.get("reasoning", "") or msg.get("reasoning_content", "")
    
    print(f"\n📝 model 返回 content:")
    print(f"   原始: {repr(content)}")
    if reasoning:
        print(f"   reasoning: {repr(reasoning[:200])}{'...' if len(reasoning) > 200 else ''}")

    # === 与 search.py 完全一致的解析逻辑 ===
    ans = content.strip()
    
    # 清除特殊标记（GLM-4 系列的 <|begin_of_box|>...<|end_of_box|> 等）
    ans = re.sub(r'<\|[^|]*\|>', '', ans).strip()
    
    # content 为空时从 reasoning 兜底提取
    if not ans and reasoning:
        all_arrays = re.findall(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', reasoning)
        if all_arrays:
            ans = all_arrays[-1]
            print(f"   ℹ️ content 为空，从 reasoning 提取: {ans}")
    
    if not ans:
        print(f"\n❌ content 和 reasoning 均未找到有效数组!")
        return
    
    # 正则提取数组
    array_match = re.search(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]', ans)
    if array_match:
        ans = array_match.group()
    else:
        ans = ans.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(ans)
        if isinstance(result, list):
            print(f"\n✅ 解析成功! 排序结果: {result}")
        else:
            print(f"\n⚠️ 解析非数组: {type(result).__name__} = {result}")
    except json.JSONDecodeError as e:
        print(f"\n❌ 最终解析失败: {e}")
        print(f"   清理后: {repr(ans)}")


def main():
    print("🧪 Ollama 本地模型文本能力测试")
    print("   模拟场景：重排序任务（与搜索功能使用完全相同的 prompt）\n")

    # 从数据库读取所有本地模型
    mgr = ModelManager(get_model_db_path("photo_identify_iPhone.db"))
    all_models = mgr.get_all_models()
    mgr.close()

    local_models = [m for m in all_models if m.get("is_local")]
    
    if not local_models:
        print("⚠️ 未找到本地模型，请先在模型管理页面添加。")
        # 也测试一下所有模型
        print("\n已注册的全部模型：")
        for m in all_models:
            api_key_str = "(本地)" if m.get("is_local") else m["api_key_var"]
            print(f"  {m['type']} | {m['name']} | {m['base_url']} | {api_key_str}")
        return

    print(f"找到 {len(local_models)} 个本地模型，开始测试...\n")

    for m in local_models:
        test_model(m["name"], m["model_id"], m["base_url"])

    # 也测试远程模型作为对比
    remote_models = [m for m in all_models if not m.get("is_local") and m["api_key_status"]]
    if remote_models:
        print(f"\n\n{'#'*60}")
        print(f"# 对比测试：远程模型（已配置 API Key 的）")
        print(f"{'#'*60}")
        for m in remote_models[:1]:  # 只测试第一个远程模型作为对比
            from photo_identify.model_manager import ModelManager as MM
            api_key = MM.get_api_key_value(m["api_key_var"])
            test_model(m["name"], m["model_id"], m["base_url"], api_key)

    print(f"\n\n{'='*60}")
    print("测试完成！")


if __name__ == "__main__":
    main()
