"""
SubAgent plugin for KiraAI
让主代理能够将任务委派给拥有独立人设、工具集和模型配置的子代理。

v1.1.0 设计要点：
- 异步派发模式：spawn_subagent 立即返回，子代理后台执行，完成后经消息缓冲
  防抖机制"尽力合并"地通知主 LLM 主动向用户汇报（或直接发用户，可配置）
- 协调者层（可选，默认关）：主 LLM 只把任务交给协调者，协调者异步并行派发
  下级子代理、用 collect_subagent_results 收集结果；防烂尾三规则
  （收尾拦截 / 级联取消 / 下级免投递）；人设注册与读取两侧与普通层完全隔离
- 任务管理：subagent_status / stop_subagent / resume_subagent + 命令双通道
- 人设系统集成：内置人设自动注册到 webui（重名不覆盖），persona_id 稳定绑定
- 工具白/黑名单（关键词匹配可开关）、可选模型列表、会话作用域、步数双层限制
"""

import asyncio
import base64
import contextlib
import contextvars
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.plugin import BasePlugin, register, on, Priority
from core.logging_manager import get_logger
from core.agent.agent_executor import AgentExecutor, AgentExecutionContext
from core.agent.tool import ToolSet, ToolResult
from core.agent.message import OpenAIMessage
from core.utils.tool_utils import BaseTool
from core.utils.path_utils import get_data_path
from core.prompt_manager import Prompt
from core.provider import LLMRequest
from core.chat.session import Group, Session
from core.chat.message_utils import KiraMessageBatchEvent, KiraMessageEvent, KiraIMMessage
from core.chat import MessageChain
from core.chat.message_elements import Text
from core.adapter.adapter_info import AdapterInfo
from core.persona.model import PersonaInfo

# 注意：颜色名必须是 colorlog 内置或框架自定义的颜色码
# （black/red/green/yellow/blue/purple/cyan/white/orange），
# 用不存在的颜色（如 magenta）会导致每条日志格式化时报 KeyError。
sub_logger = get_logger("subagent", "purple")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class SubAgentConfig:
    subagent_id: str
    name: str
    description: str
    persona: str = ""
    persona_id: str = ""          # 绑定的 webui 人设 id（稳定锚点，改名不影响）
    source: str = "builtin"       # builtin | persona | llm
    tools: list[str] = field(default_factory=list)   # 额外白名单（空 = 不限制）
    max_steps: int = 0            # 0 = 用插件默认值
    timeout: float = 0.0          # 0 = 用插件默认值
    model: str = ""               # "provider_id:model_id" / "fast" / ""
    tags: str = ""                # 擅长标签提示
    tier: str = "normal"          # normal | coordinator（协调者：管其他子代理的子代理）


@dataclass
class SubAgentTask:
    task_id: str
    subagent_id: str
    name: str
    sid: str                      # 发起会话真实 sid
    origin: str                   # tool | command
    task_text: str
    state: str = "queued"         # queued | running | done | timeout | stopped | error
    current_step: int = 0
    max_steps: int = 0
    last_step_summary: str = ""
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    result: str = ""
    error: str = ""
    handle: Optional[asyncio.Task] = None
    request: object = None        # 保留 LLMRequest 用于 resume
    llm_model: object = None
    tier: str = "normal"          # normal | coordinator（冗余自 cfg，防 cfg 先被删）
    parent_task_id: str = ""      # 派出本任务的协调者任务ID（origin=coordinator 时）
    collected: bool = False       # 结果是否已被协调者 collect 回收


# 子代理管理类工具的安全底线：任何情况下子代理自身都不能调用（防递归/防自管理）
_SOURCE_LABELS = {"builtin": "内置", "persona": "用户人设", "llm": "AI创建", "coordinator": "协调者创建"}

# 子代理最终产出里常裹着框架消息标记 <msg><text>...</text></msg>，
# 直接投递会原样显示标签，注入主 AI 也会带噪音。只剥这两个容器标签，
# 保留 <file> 等媒体标签（主 AI 汇报时可凭它发图，框架 xml_tag_fixer 也认它们）。
_MSG_TAG_RE = re.compile(r"</?msg\b[^>]*/?>|</?text\s*>", re.I)


def _clean_result_markup(text: str) -> str:
    """剥掉结果文本里的 <msg>/<text> 容器标签（可多层嵌套），其余标签保留。"""
    if not text or ("<" not in text):
        return text
    prev = None
    while prev != text:   # 多层嵌套时反复剥
        prev = text
        text = _MSG_TAG_RE.sub("", text)
    return text.strip()


# <file type="image/record/video/file">路径或URL</file>，type 可省（默认 file）
_FILE_TAG_RE = re.compile(
    r"<file(?:\s+type=\"(image|record|video|file)\")?\s*>(.*?)</file>", re.S | re.I)


_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024  # data URI 附件上限

_MIME_EXT = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/gif": ".gif", "image/webp": ".webp", "image/svg+xml": ".svg",
    "image/bmp": ".bmp",
    "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/ogg": ".ogg",
    "audio/mp4": ".m4a", "audio/amr": ".amr",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    # 常用文档
    "text/plain": ".txt", "text/markdown": ".md", "text/csv": ".csv",
    "text/html": ".html", "application/json": ".json",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/zip": ".zip", "application/x-rar-compressed": ".rar",
    "application/x-7z-compressed": ".7z",
}


def _resolve_data_uri_files(text: str, task_id: str = "") -> str:
    """<file> 标签里塞 data:...;base64,... 数据 URI 的，解码落盘到 data/temp
    并把标签改写成文件路径。框架和本插件的 <file> 解析都只认路径/URL，
    不处理数据 URI 就会整段 base64 原样刷屏；落盘后两条投递路径都能发真附件。"""
    if not text or "data:" not in text:
        return text

    def _sub(m):
        ftype = (m.group(1) or "file").lower()
        dm = re.match(r"\s*data:([\w.+-]+/[\w.+-]+);base64,(.*)$", m.group(2), re.S)
        if not dm:
            return m.group(0)
        mime = dm.group(1).lower()
        b64 = re.sub(r"\s+", "", dm.group(2))
        # 解码前先按 base64 长度估算原始大小（编码后约为原始的 4/3），
        # 避免巨大附件完整解码造成内存峰值
        if len(b64) * 3 // 4 > _MAX_ATTACHMENT_BYTES:
            return "[附件超过 25MB，已丢弃]"
        import binascii
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return "[附件 base64 数据损坏，无法还原]"
        if len(raw) > _MAX_ATTACHMENT_BYTES:
            return "[附件超过 25MB，已丢弃]"
        ext = _MIME_EXT.get(mime, ".bin")
        out = get_data_path() / "temp" / f"subagent_{task_id or 'file'}_{int(time.time() * 1000)}{ext}"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(raw)
        except OSError:
            return m.group(0)
        return f'<file type="{ftype}">data/temp/{out.name}</file>'

    return _FILE_TAG_RE.sub(_sub, text)


def _extract_file_elements(text: str):
    """把结果里的 <file> 标签解析成真实媒体元素（路径规则仿框架 kira-ai tags.py），
    使命令直发路径也能真正发出图片/文件。返回 (去掉已解析标签的文本, [元素...])。
    无法解析的标签保留原文，避免丢信息。"""
    from core.chat.message_elements import Image, File, Record, Video
    elements = []

    def _resolve(raw: str) -> str:
        v = raw.strip().replace("\\", "/")
        if not v:
            return ""
        if v.startswith(("http://", "https://")):
            return v
        p = Path(v)
        if p.is_absolute():
            return str(p) if p.exists() else ""
        if v.startswith("data/"):
            ap = get_data_path() / v.removeprefix("data/")
            return str(ap) if ap.exists() else ""
        return ""

    def _sub(m):
        ftype = (m.group(1) or "file").lower()
        resolved = _resolve(m.group(2))
        if not resolved:
            return m.group(0)
        name = None if resolved.startswith("http") else Path(resolved).name
        if ftype == "image" and resolved.lower().endswith(".svg"):
            # QQ 等客户端不支持 svg 图片渲染，降级为文件发送
            elements.append(File(file=resolved, name=name))
        elif ftype == "image":
            elements.append(Image(image=resolved, name=name))
        elif ftype == "record":
            elements.append(Record(record=resolved, name=name))
        elif ftype == "video":
            elements.append(Video(file=resolved, name=name))
        else:
            elements.append(File(file=resolved, name=name))
        return ""

    clean = _FILE_TAG_RE.sub(_sub, text)
    return clean.strip(), elements

# 子代理任务上下文标记（contextvar：异步任务间隔离，主 AI 的日志不受影响）
_IN_SUBAGENT = contextvars.ContextVar("kira_subagent_ctx", default=False)
# 当前正在执行的子代理任务（仅子代理任务协程内有值；主 LLM 上下文为 None）。
# 工具据此判断调用者是主 LLM / 协调者 / 普通子代理，实现两层层级管控。
_CURRENT_TASK: contextvars.ContextVar = contextvars.ContextVar("kira_subagent_current_task", default=None)
_QUIET_STATE = {"enabled": True}   # 由 _load_cfg 根据开关刷新
_FILTER_ATTACHED = {"done": False}


class _SubagentContextFilter(logging.Filter):
    """子代理任务上下文内屏蔽框架的过程日志（工具参数/结果、shell 命令、LLM 步骤），
    防止子代理刷屏主日志。主 AI 的消息在别的异步上下文里，不受影响。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (_QUIET_STATE["enabled"] and _IN_SUBAGENT.get(False))


def _attach_quiet_filter():
    """给框架的相关 logger 挂过滤器（幂等）。"""
    if _FILTER_ATTACHED["done"]:
        return
    for name in ("tool_use", "plugin", "llm"):
        logging.getLogger(name).addFilter(_SubagentContextFilter())
    _FILTER_ATTACHED["done"] = True

class _SubagentLLMShim:
    """包一层框架 LLMClient，仅替换 execute_tool：
    框架 core/llm_client.py 会把每次工具调用套进 wait_for(webui 工具调用超时)，
    子代理改用自己的 sub_tool_timeout，与 webui 设置解勾。
    其余属性/方法全部委托原 LLMClient（AgentExecutor 只用到 execute_tool）。"""

    def __init__(self, inner, tool_timeout: float):
        self._inner = inner
        self._tool_timeout = tool_timeout

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def execute_tool(self, event, resp, tool_set=None):
        # 0 = 跟随框架：直接走原实现（含 webui 工具调用超时）
        if self._tool_timeout <= 0:
            return await self._inner.execute_tool(event, resp, tool_set=tool_set)
        from core.plugin.plugin_handlers import event_handler_reg, EventType
        timeout = self._tool_timeout
        tool_logger = logging.getLogger("tool_use")
        for tool_call in resp.tool_calls:
            tool_call_id = tool_call.get("id")
            name = tool_call.get("function", {}).get("name")
            raw_args = tool_call.get("function", {}).get("arguments") or ""
            try:
                args = {} if not raw_args.strip() else json.loads(raw_args)
            except json.JSONDecodeError as e:
                sub_logger.error(f"子代理工具参数解析失败: {e}; 原始: {raw_args}")
                args = {}
            tool_logger.info(f"{name} args: {args}")
            if tool_set and name in tool_set:
                try:
                    coro = tool_set.get(name).execute(event, **args)
                    if name == "collect_subagent_results":
                        # collect 要阻塞等下级跑完（可能远超单次工具超时），
                        # 不受 sub_tool_timeout 约束；上限由协调者任务总超时兜底
                        result = await coro
                    else:
                        result = await asyncio.wait_for(coro, timeout)
                except asyncio.TimeoutError:
                    result = {"error": f"Tool '{name}' timed out after {timeout}s (subagent limit)"}
                    tool_logger.error(f"Tool '{name}' timed out after {timeout}s (subagent limit)")
                except Exception as e:
                    result = {"error": f"Failed to call tool '{name}': {e}"}
                    tool_logger.error(f"Failed to call tool '{name}': {e}")
            else:
                result = {"error": f"Tool {name} not implemented"}
                tool_logger.error(f"Tool {name} not implemented")
            tool_result_obj = result if isinstance(result, ToolResult) else ToolResult(str(result))
            for handler in event_handler_reg.get_handlers(event_type=EventType.ON_TOOL_RESULT):
                await handler.exec_handler(event, tool_result_obj)
                if event.is_stopped:
                    return
            content = await tool_result_obj.assemble_result()
            tool_logger.info(f"tool_result: {content}")
            resp.tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            })


# DeepSeek DSML 标记（部分代理网关会把工具调用以私有标记漏成纯文本）
_DSML_BLOCK_RE = re.compile(
    r"<[|｜]+DSML[|｜]+tool_calls>(.*?)</[|｜]+DSML[|｜]+tool_calls>", re.S)
_DSML_INVOKE_RE = re.compile(
    r"<[|｜]+DSML[|｜]+invoke\s+name=\"([^\"]+)\"[^>]*>(.*?)</[|｜]+DSML[|｜]+invoke>", re.S)
_DSML_PARAM_RE = re.compile(
    r"<[|｜]+DSML[|｜]+parameter\s+name=\"([^\"]+)\"[^>]*>(.*?)</[|｜]+DSML[|｜]+parameter>", re.S)


def _source_label(source: str) -> str:
    """子代理来源中文标签：区分内置 / 用户人设 / AI创建。"""
    return _SOURCE_LABELS.get(source, source)


_SAFETY_BOTTOM = {
    "call_subagent", "spawn_subagent", "register_subagent", "edit_subagent",
    "remove_subagent", "list_subagents", "subagent_status", "stop_subagent",
    "resume_subagent", "get_subagent_persona", "save_subagent_persona",
    "collect_subagent_results",
}

# 协调者豁免的管理工具（仍在 _SAFETY_BOTTOM 里，仅协调者 tier 放行）：
# 派/查/停/续/创建/编辑/收集。不能 remove、不能 call（阻塞自身）、
# 不能存人设、不能读人设全文；不能派协调者（spawn 里的层级拦截保证）
_COORDINATOR_TOOLS = {
    "spawn_subagent", "list_subagents", "subagent_status", "stop_subagent",
    "resume_subagent", "register_subagent", "edit_subagent", "collect_subagent_results",
}

_BUILTIN_IDS = {"subagent_code_expert", "subagent_writing_expert"}

_BUILTIN_PERSONAS = [
    {
        "key": "code_expert",
        "persona_id": "subagent_code_expert",
        "base_name": "子代理-代码专家",
        "tags": "代码 审查 重构 调试",
        "content": (
            "你是一位资深软件工程师，擅长代码审查、Bug 定位、重构和技术方案评估。"
            "优先给出可运行的代码示例，指出潜在风险，保持代码风格一致。"
        ),
    },
    {
        "key": "writing_expert",
        "persona_id": "subagent_writing_expert",
        "base_name": "子代理-写作专家",
        "tags": "写作 润色 文案",
        "content": (
            "你是一位专业的写作专家，擅长创作各类文字内容。"
            "无论是小说、散文、诗歌、剧本，还是工作报告、技术文档、广告文案，"
            "你都能根据需求完成。注重文笔流畅、逻辑清晰、风格贴合目标读者。"
        ),
    },
]

_COORDINATOR_BUILTIN_ID = "subagent_coordinator_planner"
_COORDINATOR_BUILTIN = {
    "key": "planner",
    "persona_id": _COORDINATOR_BUILTIN_ID,
    "base_name": "子代理-规划师",
    "tags": "规划 拆解 管理",
    "content": (
        "你是「规划师」，一名任务协调者。你不亲自执行具体工作，负责把复杂任务\n"
        "拆解、分派给下级子代理、跟踪进度并审查结果，直到任务真正完成。\n"
        "\n"
        "工作方式：\n"
        "1. 拆解：把收到的任务拆成若干独立的可执行单元，明确每个单元的交付标准。\n"
        "2. 分派：用工具把每个单元派给合适的下级子代理（可按需创建新的下级）。\n"
        "   派活时必须说清：要做什么、交付什么、多少步以内完成。\n"
        "3. 跟踪：用查询工具跟进进度；下级卡住或失败时调整方案重新派。\n"
        "4. 审查：关键产物（代码、文档等）应派一个未参与制作的下级做独立审查；\n"
        "   审查不通过说明原因并返工，直至达标。简单任务可省略审查。\n"
        "5. 交付：全部完成后，汇总各下级产出，输出最终完整结果。\n"
        "\n"
        "规则：\n"
        "- 只协调，不亲自动手做具体实现。\n"
        "- 不能跳过交付直接结束；任务未完成就继续安排，直到完成或确认无法完成。\n"
        "- 若确实无法推进，如实说明：已完成什么、卡在哪、需要什么。"
    ),
}

_STUB_ADAPTER = AdapterInfo(
    enabled=True,
    adapter_id="subagent",
    name="subagent",
    platform="subagent",
    description="SubAgent stub adapter",
)


class SubAgentPlugin(BasePlugin):
    """SubAgent plugin: lets the main agent delegate tasks to specialized sub-agents."""

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self._configs: dict[str, SubAgentConfig] = {}
        self._default_order: list[str] = []          # 默认子代理列表（来自 default_personas 配置）
        self._coordinator_order: list[str] = []      # 协调者列表（来自 coordinator_personas 配置）
        self._coordinator_spawned: set[str] = set()  # 协调者创建的内存级下级（不持久化）
        self._coordinator_spawned_by: dict[str, set[str]] = {}  # 协调者任务ID -> 其创建的内存级下级
        self._hot_loaded_order: list[str] = []       # 热加载的 AI 创建子代理（命令序号排在默认列表之后）
        self._tasks: dict[str, SubAgentTask] = {}
        self._task_counter = 0
        self._sem: Optional[asyncio.Semaphore] = None
        self._session_sems: dict[str, asyncio.Semaphore] = {}
        self._custom_tools_cache: Optional[list[BaseTool]] = None
        self._last_status_query: dict[str, float] = {}   # sid -> 上次查询时间
        self._store: dict = {"persona_map": {}, "saved": {}, "coordinator_persona_map": {}}
        self._store_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # 配置读取
    # ------------------------------------------------------------------

    def _load_cfg(self):
        c = self.plugin_cfg
        g = c.get("section_general", {})
        self.call_mode = g.get("call_mode", "async")
        self.result_delivery = g.get("result_delivery", "to_main_llm")
        self.merge_window_ms = int(g.get("merge_window_ms", 100))
        self.max_concurrent = int(g.get("max_concurrent", 10))
        self.max_concurrent_per_session = int(g.get("max_concurrent_per_session", 5))
        self.default_max_steps = int(g.get("default_max_steps", 20))
        self.max_steps_limit = int(g.get("max_steps_limit", 25))
        # 主 LLM 设置的步数下限（地板）：防止它保守地填 5 之类的小值。
        # 仅约束 LLM 主动给的值；填 0/不填走插件默认值不受限；webui 手动配置不受限。0=不限制
        self.min_llm_steps = int(g.get("min_llm_steps", 15))
        self.default_timeout = float(g.get("default_timeout", 300))
        self.force_final_report = bool(g.get("force_final_report", True))
        # 投递前剥掉结果里的 <msg>/<text> 容器标签（子代理 LLM 常把框架消息
        # 标记当普通文本输出，直接投递会原样显示标签）。
        # 与 xml_tag_fixer 插件互补不冲突：它修框架层 LLM 输出，这里清子代理结果
        self.strip_msg_tags = bool(g.get("strip_msg_tags", True))
        self.status_query_interval = float(g.get("status_query_interval", 20))
        self.enabled_sessions = [s.strip() for s in g.get("enabled_sessions", []) if str(s).strip()]

        self.available_models_raw = [s for s in c.get("section_models", {}).get("available_models", []) if str(s).strip()]
        self.available_models = self._parse_model_list(self.available_models_raw)

        t = c.get("section_tools", {})
        self.tool_whitelist = [s.strip() for s in t.get("tool_whitelist", []) if str(s).strip()]
        # 默认黑名单不含 exec：exec 是否可用完全交给官方文件插件管控
        # （allowed_exec_sessions + exec_deny_list），避免双重封禁
        self.tool_blacklist = [s.strip() for s in t.get("tool_blacklist", [
            "send_email", "mijia_control_device", "delete_qq_msg",
            "qzone_delete", "qzone_publish",
        ]) if str(s).strip()]
        self.tool_match_fuzzy = bool(t.get("tool_match_fuzzy", True))
        self.allowed_read_paths = self._as_list(
            t.get("allowed_read_paths"), ["data/files", "data/temp", "data/plugins"])
        self.allowed_write_paths = self._as_list(
            t.get("allowed_write_paths"), ["data/files", "data/temp"])
        self.inject_framework_brief = bool(t.get("inject_framework_brief", True))
        # exec 非阻塞包装：文件插件 exec 用同步 subprocess，会卡住整个事件循环。
        # 开（默认）：子代理的 exec 放到工作线程执行，主进程不被阻塞
        self.exec_non_blocking = bool(t.get("exec_non_blocking", True))
        # DSML 补执行：部分 DeepSeek 代理会把工具调用漏成 DSML 纯文本。
        # 开（默认）：解析这些标记并真正执行对应工具，防止任务"假完成"
        self.dsml_rescue = bool(t.get("dsml_rescue", True))
        # 子代理单次工具调用超时（秒），与 webui 框架「工具调用超时」解勾。
        # 框架 core/llm_client.py 会把每次工具调用套进 wait_for(tool_call_timeout)，
        # 这里用插件自己的值替代；0 = 跟随框架 webui 设置
        self.sub_tool_timeout = float(t.get("sub_tool_timeout", 120))

        p = c.get("section_persona", {})
        self.default_personas_raw = [s for s in p.get("default_personas", [
            "子代理-代码专家;代码 审查 重构 调试",
            "子代理-写作专家;写作 润色 文案",
        ]) if str(s).strip()]
        self.allow_llm_read_persona = bool(p.get("allow_llm_read_persona", False))
        self.keyword_persona_excerpt = bool(p.get("keyword_persona_excerpt", False))
        self.persona_excerpt_length = int(p.get("persona_excerpt_length", 120))
        self.allow_llm_create = bool(p.get("allow_llm_create_subagent", True))
        self.allow_llm_save = bool(p.get("allow_llm_save_persona", True))
        self.llm_decide_save = bool(p.get("llm_decide_save_persona", True))
        self.hot_reload_saved = bool(p.get("hot_reload_saved_persona", True))

        r = c.get("section_resume", {})
        self.allow_resume = bool(r.get("allow_resume", True))
        self.resume_keep_minutes = int(r.get("resume_keep_minutes", 30))

        co = c.get("section_coordinator", {})
        # 协调者总开关（默认关）：开启后主 LLM 只把任务交给协调者，由它分派下级子代理
        self.enable_coordinator = bool(co.get("enable_coordinator", False))
        self.coordinator_personas_raw = [s for s in co.get("coordinator_personas", [
            "子代理-规划师;规划 拆解 管理",
        ]) if str(s).strip()]
        # 协调者可用模型列表（格式同 available_models；留空回退 available_models）
        self.coordinator_models_raw = [s for s in co.get("coordinator_models", []) if str(s).strip()]
        self.coordinator_models = self._parse_model_list(self.coordinator_models_raw)
        self.coordinator_default_steps = int(co.get("coordinator_default_steps", 25))
        # 协调者步数硬上限独立于全局 max_steps_limit（全局那个只约束下级子代理）
        self.coordinator_max_steps_limit = int(co.get("coordinator_max_steps_limit", 50))
        self.coordinator_timeout = float(co.get("coordinator_timeout", 900))
        self.coordinator_context_minutes = int(co.get("coordinator_context_minutes", 60))
        self.coordinator_save_spawned = bool(co.get("coordinator_save_spawned", False))

        cmd = c.get("section_command", {})
        self.enable_commands = bool(cmd.get("enable_commands", False))
        self.cmd_start_aliases = [s.strip() for s in cmd.get("cmd_start_aliases", ["/suba", "/子代理"]) if str(s).strip()]
        self.cmd_stop_aliases = [s.strip() for s in cmd.get("cmd_stop_aliases", ["/stopsuba", "/停止子代理"]) if str(s).strip()]
        self.cmd_resume_aliases = [s.strip() for s in cmd.get("cmd_resume_aliases", ["/resumesuba", "/继续子代理"]) if str(s).strip()]
        self.cmd_result_to_main_llm = bool(cmd.get("cmd_result_to_main_llm", False))
        self.stop_return_progress = bool(cmd.get("stop_return_progress", True))
        # 命令白名单：留空=所有人可用；填了=只有名单内 QQ 号可用（仅约束命令通道，不影响主 AI 工具调用）
        self.cmd_allowed_users = [str(u).strip() for u in self._as_list(cmd.get("cmd_allowed_users"), []) if str(u).strip()]
        # 命令序号是否包含热加载的 AI 子代理（默认开：热加载开启时它们也按序号可调）
        self.cmd_include_hot_loaded = bool(cmd.get("cmd_include_hot_loaded", True))

        # 回复语模版（独立分组 section_messages；兼容旧配置 section_command 里的同名字段）
        ms = c.get("section_messages", {})
        def _msg(key, default):
            return ms.get(key, cmd.get(key, default))
        self.cmd_denied_message = _msg("cmd_denied_message", "⛔ 你没有使用子代理命令的权限。")
        self.msg_no_default = _msg("msg_no_default", "尚未设置默认的子代理，请在插件设置的 default_personas 中配置。")
        self.msg_invalid_index = _msg("msg_invalid_index", "无效的子代理序号 {index}。当前默认子代理：\n{list}")
        self.msg_started = _msg("msg_started", "已派出子代理「{name}」处理任务（任务ID {task_id}），完成后我会告诉你。")
        self.msg_stopped = _msg("msg_stopped", "已停止子代理任务 {task_id}（{name}）。")
        self.msg_none_running = _msg("msg_none_running", "当前没有正在运行的子代理任务。")
        self.cmd_result_template = _msg("cmd_result_template", "【子代理 {name}】任务完成：\n{result}")
        self.notify_template = _msg(
            "notify_template",
            "【系统通知】你派出的子代理「{name}」（任务ID {task_id}）已完成任务「{task}」，结果如下。请以自己的口吻向用户汇报：\n{result}",
        )
        # 收尾汇报调用的超时。收尾请求带完整任务上下文（token 量大），
        # 慢服务商 30 秒容易不够，默认 60
        self.wrapup_timeout = float(g.get("wrapup_timeout", 60))

        # 调试开关：总开关 + 三个细分内容开关（细分开关需总开关开启才生效）
        dbg = c.get("section_debug", {})
        self.debug_enabled = bool(dbg.get("debug_enabled", False))
        self.debug_log_prompts = self.debug_enabled and bool(dbg.get("debug_log_prompts", False))
        self.debug_log_steps = self.debug_enabled and bool(dbg.get("debug_log_steps", False))
        self.debug_log_results = self.debug_enabled and bool(dbg.get("debug_log_results", False))
        # 静默子代理过程日志（默认开）：屏蔽框架在子代理上下文里打的工具/LLM 步骤日志。
        # 调试模式开启时自动失效（全部显示，方便排查）
        self.quiet_subagent_logs = bool(dbg.get("quiet_subagent_logs", True))
        _QUIET_STATE["enabled"] = self.quiet_subagent_logs and not self.debug_enabled
        self._dbg("配置已加载: call_mode=%s, 并发=%s/%s, 步数=%s(上限%s), 命令=%s, 白名单=%s, 调试=%s",
                  self.call_mode, self.max_concurrent, self.max_concurrent_per_session,
                  self.default_max_steps, self.max_steps_limit,
                  "开" if self.enable_commands else "关",
                  self.cmd_allowed_users or "所有人", self.debug_enabled)

    @staticmethod
    def _as_list(value, default: list) -> list:
        """兼容 list 配置与旧的逗号分隔 string 配置。"""
        if value is None:
            return list(default)
        if isinstance(value, list):
            return [str(s).strip() for s in value if str(s).strip()]
        # 旧格式：逗号分隔字符串
        return [s.strip() for s in str(value).split(",") if s.strip()]

    @staticmethod
    def _parse_model_list(raw_list) -> list[dict]:
        """解析 'provider;model;提示(可空)' 列表。"""
        out = []
        for line in raw_list:
            parts = [p.strip() for p in str(line).split(";")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                sub_logger.warning(f"[subagent] available_models 条目格式错误（需要 提供商;模型）: {line}")
                continue
            out.append({"provider": parts[0], "model": parts[1], "hint": parts[2] if len(parts) > 2 else ""})
        return out

    # ------------------------------------------------------------------
    # 持久化 store（persona_id 映射 + LLM 保存的子代理）
    # ------------------------------------------------------------------

    def _store_load(self):
        try:
            data_dir = self.ctx.get_plugin_data_dir()
            if data_dir:
                self._store_path = data_dir / "subagents.json"
                if self._store_path.exists():
                    self._store = json.loads(self._store_path.read_text(encoding="utf-8"))
                    self._store.setdefault("persona_map", {})
                    self._store.setdefault("saved", {})
                    self._store.setdefault("coordinator_persona_map", {})
        except Exception as e:
            sub_logger.error(f"[subagent] 读取 subagents.json 失败: {e}")

    def _store_save(self):
        if not self._store_path:
            return
        try:
            self._store_path.write_text(
                json.dumps(self._store, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            sub_logger.error(f"[subagent] 写入 subagents.json 失败: {e}")

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def _migrate_legacy_timeouts(self):
        """旧时代存档迁移：早期版本默认总超时是 120，注册/编辑时被显式写进
        subagents.json（timeout: 120），导致用户后来改了默认超时这些存档也不跟进。
        与步数地板迁移同理：把恰好等于旧默认 120 的存档超时改为 0（跟随插件默认值）。"""
        # 只迁移一次：否则用户在迁移后故意把超时设回 120，每次重启都会被重置
        if self._store.get("_migration_version", 0) >= 1:
            return
        migrated = []
        for said, rec in self._store.get("saved", {}).items():
            try:
                if float(rec.get("timeout", 0.0)) == 120.0:
                    rec["timeout"] = 0.0
                    migrated.append(said)
            except (TypeError, ValueError):
                continue
        self._store["_migration_version"] = 1
        self._store_save()
        if migrated:
            sub_logger.info(
                f"[subagent] 旧默认超时迁移：{', '.join(migrated)} 的存档超时 120s → 跟随插件默认值")

    async def initialize(self):
        self._load_cfg()
        _attach_quiet_filter()
        self._store_load()
        self._migrate_legacy_timeouts()
        self._sem = asyncio.Semaphore(self.max_concurrent)

        await self._ensure_builtin_personas()
        await self._load_default_personas()
        await self._load_coordinator_personas()
        await self._load_saved_subagents()

        sub_logger.info(
            f"SubAgent plugin loaded. subagents={list(self._configs.keys())}, "
            f"default_order={self._default_order}, coordinators={self._coordinator_order}, "
            f"coordinator={'开' if self.enable_coordinator else '关'}, "
            f"mode={self.call_mode}/{self.result_delivery}"
        )

    async def terminate(self):
        for t in self._tasks.values():
            if t.handle and not t.handle.done():
                t.handle.cancel()
        self._configs.clear()
        self._default_order.clear()
        self._coordinator_order.clear()
        self._coordinator_spawned.clear()
        self._tasks.clear()
        self._last_status_query.clear()
        self._custom_tools_cache = None
        sub_logger.info("SubAgent plugin terminated")

    def register_subagent_config(self, config: SubAgentConfig):
        """Public API for other plugins to register their own sub-agents."""
        self._configs[config.subagent_id] = config

    # ------------------------------------------------------------------
    # 人设系统集成
    # ------------------------------------------------------------------

    async def _ensure_builtin_personas(self):
        """把内置子代理人设注册到 webui 人设库。重名不覆盖；已删除不重建（策略A）。"""
        persona_mgr = self.ctx.persona_mgr
        try:
            existing = await persona_mgr.list_personas()
        except Exception as e:
            sub_logger.error(f"[subagent] 无法列出人设: {e}")
            return
        existing_names = {p.name for p in existing}

        # 普通内置人设记入 persona_map；协调者（规划师）记入独立的
        # coordinator_persona_map，与普通子代理层在注册/读取两侧完全隔离。
        # 协调者人设无论开关与否都注册：开关只控制可用性，用户可在 webui 看到并修改
        for bp, pmap in [(b, self._store["persona_map"]) for b in _BUILTIN_PERSONAS] \
                + [(_COORDINATOR_BUILTIN, self._store.setdefault("coordinator_persona_map", {}))]:
            pid = bp["persona_id"]
            pmap.setdefault(bp["key"], pid)
            cur = await persona_mgr.get_persona(pid)
            if cur:
                # 已存在（含用户改名/改内容的情况）→ 直接使用，绝不覆盖
                continue
            if self._store.get("builtin_created", {}).get(pid):
                # 曾经创建过但库里没有 → 用户主动删除 → 尊重删除，不重建
                sub_logger.warning(
                    f"[subagent] 内置子代理人设 {pid}（{bp['base_name']}）已被删除，"
                    f"不再自动重建。如需恢复请在 webui 手动新建人设并配置到 default_personas。"
                )
                continue
            # 首次安装：挑一个不冲突的名字
            name = bp["base_name"]
            i = 2
            while name in existing_names:
                name = f"{bp['base_name']}-{i}"
                i += 1
            try:
                await persona_mgr.create_persona(PersonaInfo(
                    id=pid, name=name, format="text", content=bp["content"],
                ))
                existing_names.add(name)
                pmap[name] = pid  # 记录 创建名→id，后续改名也不影响解析
                self._store.setdefault("builtin_created", {})[pid] = True
                sub_logger.info(f"[subagent] 已注册内置人设: {pid}（{name}）")
            except Exception as e:
                sub_logger.error(f"[subagent] 创建内置人设 {pid} 失败: {e}")
        self._store_save()

    async def _load_default_personas(self):
        """按 default_personas 配置（人设名;标签）加载默认可用子代理。"""
        persona_mgr = self.ctx.persona_mgr
        pmap = self._store["persona_map"]
        try:
            all_personas = await persona_mgr.list_personas()
        except Exception as e:
            sub_logger.error(f"[subagent] 无法列出人设: {e}")
            return
        by_name = {p.name: p for p in all_personas}
        by_id = {p.id: p for p in all_personas}

        for line in self.default_personas_raw:
            parts = [s.strip() for s in str(line).split(";")]
            pname, tags = parts[0], (parts[1] if len(parts) > 1 else "")
            if not pname:
                continue
            # 解析顺序：缓存的 persona_id（改名不影响）→ id 精确匹配 →
            # 内置人设 base_name 兜底（防同名人设抢走内置绑定）→ 名称匹配
            target = None
            cached_id = pmap.get(pname)
            if cached_id and cached_id in by_id:
                target = by_id[cached_id]
            elif pname in by_id:
                target = by_id[pname]
            else:
                for bp in _BUILTIN_PERSONAS:
                    if pname == bp["base_name"] and bp["persona_id"] in by_id:
                        target = by_id[bp["persona_id"]]
                        break
                if not target and pname in by_name:
                    target = by_name[pname]
                    pmap[pname] = target.id  # 缓存解析结果
                    self._store_save()
            if not target:
                sub_logger.warning(f"[subagent] default_personas 中的人设「{pname}」不存在，已跳过")
                continue
            if target.id == _COORDINATOR_BUILTIN_ID:
                # 隔离：协调者人设不会被普通层 default_personas 拉走
                sub_logger.warning(
                    f"[subagent] default_personas 中的「{pname}」是协调者人设，"
                    f"请改配到 coordinator_personas，已跳过")
                continue

            sid_ = target.id  # subagent_id 直接使用 persona_id，稳定且唯一
            saved_override = self._store["saved"].get(sid_, {})
            cfg = SubAgentConfig(
                subagent_id=sid_,
                name=target.name or pname,
                description=saved_override.get("description") or (f"擅长：{tags}" if tags else f"基于人设「{target.name}」的子代理"),
                persona=target.content or "",
                persona_id=target.id,
                source="builtin" if target.id in _BUILTIN_IDS else "persona",
                tools=saved_override.get("tools", []),
                max_steps=int(saved_override.get("max_steps", 0)),
                timeout=float(saved_override.get("timeout", 0.0)),
                model=saved_override.get("model", ""),
                tags=tags,
            )
            # 旧 override 迁移：地板功能上线前存下的小步数，加载时抬到下限
            if cfg.max_steps > 0 and self.min_llm_steps > 0:
                cfg.max_steps = min(max(cfg.max_steps, self.min_llm_steps), self.max_steps_limit)
            # 人设内容以 webui 实时内容为准，覆盖保存的旧文本
            self._configs[sid_] = cfg
            if sid_ not in self._default_order:
                self._default_order.append(sid_)

    async def _load_coordinator_personas(self):
        """按 coordinator_personas 配置加载协调者（独立于默认子代理列表）。
        开关关闭时不加载：协调者人设即使在 webui 人设库里也不出现在任何可用列表。"""
        if not self.enable_coordinator:
            return
        persona_mgr = self.ctx.persona_mgr
        pmap = self._store.setdefault("coordinator_persona_map", {})
        try:
            all_personas = await persona_mgr.list_personas()
        except Exception as e:
            sub_logger.error(f"[subagent] 无法列出人设: {e}")
            return
        by_name = {p.name: p for p in all_personas}
        by_id = {p.id: p for p in all_personas}

        for line in self.coordinator_personas_raw:
            parts = [s.strip() for s in str(line).split(";")]
            pname, tags = parts[0], (parts[1] if len(parts) > 1 else "")
            if not pname:
                continue
            # 解析顺序与普通层一致（缓存id → id匹配 → 名称匹配），
            # 内置兜底只认规划师（协调者层专属，不吃普通层的内置人设）
            target = None
            cached_id = pmap.get(pname)
            if cached_id and cached_id in by_id:
                target = by_id[cached_id]
            elif pname in by_id:
                target = by_id[pname]
            else:
                if pname == _COORDINATOR_BUILTIN["base_name"] and _COORDINATOR_BUILTIN_ID in by_id:
                    target = by_id[_COORDINATOR_BUILTIN_ID]
                if not target and pname in by_name:
                    target = by_name[pname]
                    pmap[pname] = target.id  # 缓存解析结果
                    self._store_save()
            if not target:
                sub_logger.warning(f"[subagent] coordinator_personas 中的人设「{pname}」不存在，已跳过")
                continue

            sid_ = target.id
            if sid_ in self._default_order:
                # 同人设同时配在 default_personas 和 coordinator_personas：
                # 保留普通子代理身份（先加载的优先），协调者层跳过并警告，避免身份被悄悄改写
                sub_logger.warning(
                    f"[subagent] 「{pname}」同时出现在 default_personas 和 coordinator_personas，"
                    f"已按普通子代理保留，协调者层跳过")
                continue
            saved_override = self._store["saved"].get(sid_, {})
            cfg = SubAgentConfig(
                subagent_id=sid_,
                name=target.name or pname,
                description=saved_override.get("description") or (f"擅长：{tags}" if tags else f"基于人设「{target.name}」的协调者"),
                persona=target.content or "",
                persona_id=target.id,
                source="builtin" if sid_ == _COORDINATOR_BUILTIN_ID else "persona",
                tools=saved_override.get("tools", []),
                max_steps=int(saved_override.get("max_steps", 0)),
                timeout=float(saved_override.get("timeout", 0.0)),
                model=saved_override.get("model", ""),
                tags=tags,
                tier="coordinator",
            )
            self._configs[sid_] = cfg
            if sid_ not in self._coordinator_order:
                self._coordinator_order.append(sid_)
        if not self._coordinator_order:
            sub_logger.warning(
                "[subagent] 协调者已启用但 coordinator_personas 没有可用项，主 LLM 将无协调者可派")

    async def _load_saved_subagents(self):
        """恢复 LLM 创建并保存的子代理（持久化在 store 中）。绑定人设的以 webui 实时内容为准。"""
        migrated = False
        for said, rec in self._store["saved"].items():
            if said in self._configs:
                continue
            # 隔离：协调者的存档/覆盖记录不进入普通子代理层；
            # 协调者开关关闭时其记录也不恢复（整层不可见）
            if rec.get("tier") == "coordinator" or said == _COORDINATOR_BUILTIN_ID:
                continue
            persona_text = rec.get("persona", "")
            pid = rec.get("persona_id", "")
            if pid:
                try:
                    p = await self.ctx.persona_mgr.get_persona(pid)
                    if p:
                        persona_text = p.content or persona_text
                except Exception:
                    pass
            cfg = SubAgentConfig(
                subagent_id=said,
                name=rec.get("name", said),
                description=rec.get("description", ""),
                persona=persona_text,
                persona_id=pid,
                source=rec.get("source", "llm"),
                tools=rec.get("tools", []),
                max_steps=int(rec.get("max_steps", 0)),
                timeout=float(rec.get("timeout", 0.0)),
                model=rec.get("model", ""),
                tags=rec.get("tags", ""),
            )
            # 历史存档迁移：地板功能上线前 LLM 存下的小步数（如 5），加载时抬到下限
            if cfg.max_steps > 0 and self.min_llm_steps > 0:
                lifted = min(max(cfg.max_steps, self.min_llm_steps), self.max_steps_limit)
                if lifted != cfg.max_steps:
                    self._dbg("存档子代理 %s 步数 %s → 抬到下限 %s", said, cfg.max_steps, lifted)
                    cfg.max_steps = lifted
                    rec["max_steps"] = lifted
                    migrated = True
            self._configs[said] = cfg
            if cfg.persona_id and self.hot_reload_saved and said not in self._hot_loaded_order and said not in self._default_order:
                self._hot_loaded_order.append(said)
        if migrated:
            sub_logger.info("[subagent] 已将存档子代理的过小步数抬到下限，并写回持久化存储")
            self._store_save()

    def _persist_saved(self, cfg: SubAgentConfig):
        """把子代理配置持久化到 store（LLM 创建/编辑的，重启不丢）。"""
        self._store["saved"][cfg.subagent_id] = {
            "name": cfg.name,
            "description": cfg.description,
            "persona": cfg.persona,
            "persona_id": cfg.persona_id,
            "source": cfg.source,
            "tools": cfg.tools,
            "max_steps": cfg.max_steps,
            "timeout": cfg.timeout,
            "model": cfg.model,
            "tags": cfg.tags,
            "tier": cfg.tier,
        }
        self._store_save()

    # ------------------------------------------------------------------
    # 会话作用域 / 模型解析 / 工具过滤
    # ------------------------------------------------------------------

    def _dbg(self, msg: str, *args):
        """调试日志：仅调试总开关开启时输出。"""
        if self.debug_enabled:
            sub_logger.info("[调试] " + (msg % args if args else msg))

    def _in_scope(self, sid: str) -> bool:
        ok = not self.enabled_sessions or sid in self.enabled_sessions
        if not ok:
            self._dbg("会话 %s 不在启用名单内，已忽略", sid)
        return ok

    def _models_hint_text(self, tier: str = "normal") -> str:
        entries = self.coordinator_models if (tier == "coordinator" and self.coordinator_models) else self.available_models
        if not entries:
            return f"fast（webui 快速模型: {self._fast_model_label()}）"
        lines = [f"{m['provider']}:{m['model']}" + (f"（擅长: {m['hint']}）" if m["hint"] else "")
                 for m in entries]
        return "; ".join(lines)

    def _fast_model_label(self) -> str:
        """当前 webui 快速模型的真实标识（provider;model，与 available_models 行格式一致）。"""
        try:
            client = self.ctx.get_default_fast_llm_client()
            m = getattr(client, "model", None)
            if m is not None:
                provider = getattr(m, "provider_name", "") or getattr(m, "provider_id", "")
                model_id = getattr(m, "model_id", "")
                if provider and model_id:
                    return f"{provider};{model_id}"
        except Exception:
            pass
        return "（无法识别，请到 webui 模型配置里查看）"

    def _resolve_provider_inst(self, provider_key: str, model_id: str):
        """把服务商标识解析成真实 provider 实例 ID（UUID）。

        用户配置里只能也只会填人类可读的服务商显示名（如 ``Agnes（CN）``），
        而 KiraAI 的 ``get_llm_client`` 需要的是实例 UUID（如 ``31f447c5e3e3``）。
        这里自动完成映射，用户无需也无法手工抄 UUID：
        1. 直查：填的本来就是 UUID/实例 ID（老配置兼容）；
        2. 显示名精确匹配；
        3. 归一化（去空白）后精确匹配；
        4. 包含匹配（多候选取名字最短的，避免「小鸡」「小鸡（按量3）」误配）。
        每步都校验目标实例的 LLM 模型列表里确实存在该模型。
        返回 (provider_id | None, error_msg | None)。"""
        try:
            pm = self.ctx.provider_mgr
        except Exception as e:
            sub_logger.error(f"[subagent] 无法访问 provider 管理器: {e}")
            return None, "无法访问模型管理器，请检查框架状态"
        if not provider_key:
            return None, "提供商为空"
        # 1) 直查：填的已是真实实例 ID（UUID）
        inst = pm.get_provider(provider_key)
        if inst is not None:
            if pm.get_model_info(provider_key, model_id, "llm") is not None:
                return provider_key, None
            return None, f"模型 {model_id} 不在服务商 {provider_key} 的 LLM 模型列表里（请到 webui 模型配置确认）"
        # 2) 显示名匹配：精确 → 去空白精确 → 包含（候选按得分+名字长度排序，最短者胜出）
        key_norm = re.sub(r"\s+", "", provider_key)
        cands = []
        try:
            providers = pm.get_all_providers()
        except Exception:
            providers = {}
        for pid, prov in providers.items():
            name = getattr(prov, "provider_name", "") or ""
            if not name:
                continue
            if name == provider_key:
                score = 0
            else:
                name_norm = re.sub(r"\s+", "", name)
                if name_norm == key_norm:
                    score = 1
                elif key_norm and (key_norm in name_norm or name_norm in key_norm):
                    score = 2
                else:
                    continue
            if pm.get_model_info(pid, model_id, "llm") is not None:
                cands.append((score, len(name), pid))
        if cands:
            cands.sort(key=lambda x: (x[0], x[1]))
            return cands[0][2], None
        names = "、".join(sorted({
            (getattr(p, "provider_name", "") or "") for p in providers.values()
            if getattr(p, "provider_name", "")
        }))
        return None, (f"找不到服务商「{provider_key}」（或其下没有模型 {model_id}）。"
                      f"请到 webui 模型配置里填写显示名，当前已注册: {names or '（无）'}")

    def _resolve_model(self, model_str: str, tier: str = "normal"):
        """返回 (LLMModelClient | None, error_msg | None)。
        协调者优先用独立的 coordinator_models（未填则回退 available_models）。
        服务商支持 webui 显示名（自动解析成真实实例 UUID）或实例 UUID，两种都认。"""
        entries = self.coordinator_models if (tier == "coordinator" and self.coordinator_models) else self.available_models
        if not model_str:
            if not entries:
                client = self.ctx.get_default_fast_llm_client()
                return (client, None) if client else (None, "未配置快速模型")
            model_str = f"{entries[0]['provider']}:{entries[0]['model']}"
        if model_str == "fast":
            if not entries:
                client = self.ctx.get_default_fast_llm_client()
                return (client, None) if client else (None, "未配置快速模型")
            return None, (f"可选模型列表已填写，'fast' 不再可用。如需使用快速模型，"
                          f"请在 available_models 中按同格式手动添加一行：{self._fast_model_label()};擅长提示")
        provider_key, _, model_id = model_str.partition(":")
        if not provider_key or not model_id:
            return None, f"模型格式错误：{model_str}（需要 提供商;模型 或 提供商:模型）"
        # 显示名 → 真实实例 UUID（核心修复：旧版直接拿显示名查 get_llm_client 必然失败）
        real_pid, err = self._resolve_provider_inst(provider_key, model_id)
        if err:
            return None, err
        # 白名单校验：显示名与真实 UUID 命中其一即可（兼容填 UUID 的旧配置）
        if entries and not any(
                m["model"] == model_id and (m["provider"] == provider_key or m["provider"] == real_pid)
                for m in entries):
            return None, f"模型 {provider_key}:{model_id} 不在可选列表中。可选: {self._models_hint_text(tier)}"
        client = self.ctx.get_llm_client(model_uuid=f"{real_pid}:{model_id}")
        if not client:
            return None, f"无法获取模型 {provider_key}:{model_id}（服务商 {real_pid} 未注册或模型不存在）"
        return client, None

    def _name_in(self, name: str, patterns: list[str]) -> bool:
        for p in patterns:
            p = p.strip()
            if not p:
                continue
            if self.tool_match_fuzzy:
                if p in name:
                    return True
            elif p == name:
                return True
        return False

    def _sub_tool_allowed(self, name: str, cfg: "SubAgentConfig | None" = None) -> bool:
        if name in _SAFETY_BOTTOM:
            # 协调者豁免部分管理工具（派/查/停/续/创建/编辑/收集）；
            # remove / call / 存人设 / 读人设全文 仍全禁
            if not (cfg is not None and cfg.tier == "coordinator" and name in _COORDINATOR_TOOLS):
                return False
        if self.tool_whitelist and not self._name_in(name, self.tool_whitelist):
            return False
        if self.tool_blacklist and self._name_in(name, self.tool_blacklist):
            return False
        return True

    def _wrap_non_blocking(self, tool: BaseTool) -> BaseTool:
        """把会阻塞事件循环的工具（文件插件 exec 内部用同步 subprocess.run，
        会卡住整个 KiraAI 进程）包装到工作线程中执行，主事件循环不再被阻塞。
        权限判断仍在原工具内完成（用真实会话身份），安全语义不变。"""
        inner = tool

        async def _execute(self, *args, **kwargs):
            def _run():
                # 在工作线程里跑独立事件循环，避免阻塞主循环
                return asyncio.run(inner.execute(*args, **kwargs))
            return await asyncio.to_thread(_run)

        proxy_cls = type(
            f"NonBlocking_{getattr(inner, 'name', 'tool')}",
            (BaseTool,),
            {
                "name": getattr(inner, "name", None),
                "description": getattr(inner, "description", ""),
                "parameters": getattr(inner, "parameters", None),
                "execute": _execute,
            },
        )
        return proxy_cls()

    def _build_tool_set(self, cfg: SubAgentConfig) -> ToolSet:
        full_set = self.ctx.tool_mgr.build_tool_set()
        tool_set = ToolSet()
        for tool in full_set.tools:
            if not self._sub_tool_allowed(tool.name, cfg):
                continue
            if cfg.tools and not self._name_in(tool.name, cfg.tools):
                continue
            if tool.name == "exec" and self.exec_non_blocking:
                tool = self._wrap_non_blocking(tool)
            tool_set.add(tool)
        for ct in self._custom_tools():
            if self._sub_tool_allowed(ct.name, cfg) and (not cfg.tools or self._name_in(ct.name, cfg.tools)):
                tool_set.add(ct)
        return tool_set

    # ------------------------------------------------------------------
    # 子代理专用文件工具（安全路径校验）
    # ------------------------------------------------------------------

    def _resolve_safe_path(self, path: str, config_key: str) -> Optional[str]:
        """解析路径并校验是否在允许目录内。防 '..' 穿越与前缀误匹配。返回绝对路径或 None。"""
        fallback = self.allowed_read_paths if config_key == "allowed_read_paths" else self.allowed_write_paths
        allowed = self._as_list(self.plugin_cfg.get("section_tools", {}).get(config_key), fallback)
        base = get_data_path().parent  # KiraAI 根目录（data/ 的上一级）
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        try:
            rp = p.resolve()
        except Exception:
            return None
        for a in allowed:
            ap = Path(a)
            if not ap.is_absolute():
                ap = base / ap
            try:
                ra = ap.resolve()
            except Exception:
                continue
            if rp == ra or ra in rp.parents:
                return str(rp)
        return None

    def _custom_tools(self) -> list[BaseTool]:
        if self._custom_tools_cache is not None:
            return self._custom_tools_cache

        plugin = self

        class SubReadTool(BaseTool):
            name = "sub_read_file"
            description = "Read a file from allowed paths. Configure allowed_read_paths in plugin settings."
            parameters = {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path to read"}},
                "required": ["path"],
            }

            async def execute(self, event, path: str) -> str:
                rp = plugin._resolve_safe_path(path, "allowed_read_paths")
                if not rp:
                    return f"Error: path not allowed: {path}"
                try:
                    return Path(rp).read_text(encoding="utf-8")
                except Exception as e:
                    return f"Error reading file: {e}"

        class SubWriteTool(BaseTool):
            name = "sub_write_file"
            description = "Write content to a file in allowed paths. Configure allowed_write_paths in plugin settings."
            parameters = {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            }

            async def execute(self, event, path: str, content: str) -> str:
                rp = plugin._resolve_safe_path(path, "allowed_write_paths")
                if not rp:
                    return f"Error: path not allowed: {path}"
                try:
                    Path(rp).parent.mkdir(parents=True, exist_ok=True)
                    Path(rp).write_text(content, encoding="utf-8")
                    return f"Written to {rp}"
                except Exception as e:
                    return f"Error writing file: {e}"

        self._custom_tools_cache = [SubReadTool(), SubWriteTool()]
        return self._custom_tools_cache

    # ------------------------------------------------------------------
    # 执行引擎
    # ------------------------------------------------------------------

    def _make_stub_event(self, sid: str) -> KiraMessageBatchEvent:
        """构造子代理用的伪事件。关键：携带发起会话的真实 sid，
        使 file 插件的 allowed_exec_sessions / 其他按会话作用域的管控对子代理
        与主 AI 语义一致。

        ⚠ 兼容性（v1.1.3）：本事件会被广播给**所有**插件的 ON_TOOL_RESULT /
        ON_LLM_RESPONSE 钩子，而它携带的是**真实会话 sid**、又**不会**经过框架的
        ON_LLM_REQUEST（那个钩子只有 handle_im_batch_message 会派发）。凡是按会话
        记账/判态的插件（例：聊天插件的 on_llm_response 存在感统计、Midflight 的
        「运行中」状态机）都可能把子代理的输出误当成主 AI 的一轮。为此本事件带
        **显式标记**，第三方插件可直接据此跳过：
            is_stub = getattr(event, "is_subagent_stub", False) or (getattr(event, "extra", None) or {}).get("_subagent_stub")
        """
        try:
            adapter_name, st, sess_id = sid.split(":", 2)
        except ValueError:
            adapter_name, st, sess_id = "subagent", "dm", sid
        session = Session(adapter_name=adapter_name, session_type=st, session_id=sess_id)
        # 放一条合成消息：框架的 ON_LLM_RESPONSE 钩子会把本事件广播给所有插件，
        # 不少插件直接访问 event.messages[-1]（如 is_group_message()），空列表会 IndexError
        # 群会话带 Group，否则下游插件会把群来源任务误判成私聊走错分支
        dummy_msg = KiraIMMessage(
            message_id="subagent_stub", self_id="subagent",
            chain=MessageChain([]), timestamp=int(time.time()),
            session=session, group=Group(group_id=sess_id) if st == "gm" else None,
        )
        stub_event = KiraMessageBatchEvent(
            message_types=[],
            timestamp=int(time.time()),
            session=session,
            adapter=_STUB_ADAPTER,
            messages=[dummy_msg],
            # 显式标记（见 docstring「兼容性」）：让按会话记账/判态的第三方插件
            # 能一键识别「这不是主 AI 的一轮」，从而跳过本事件
            extra={"_subagent_stub": True},
        )
        try:
            stub_event.is_subagent_stub = True   # 属性通道（双保险）
        except Exception:
            pass
        return stub_event

    def _subagent_brief(self, tool_set: ToolSet, cfg: "SubAgentConfig | None" = None) -> str:
        if cfg is not None and cfg.tier == "coordinator":
            base = (
                "你是一名协调者(coordinator)，负责拆解复杂任务并分派给下级子代理完成，"
                "不亲自做具体实现。工作方式："
                "用 spawn_subagent 异步派出下级（立即返回任务ID；一步内可连发多个，下级并行运行），"
                "随后用 collect_subagent_results 等待并一次性收集它们的结果；"
                "可用 subagent_status / stop_subagent / resume_subagent 跟踪管理，"
                "可用 register_subagent / edit_subagent 创建或调整下级。"
                "你不能派出协调者，不能删除子代理。"
                "所有下级交付前你不能结束任务（系统会拦截）；全部完成后汇总输出最终结果，"
                "不要使用 XML 标签或消息格式（唯一例外：生成的文件/图片可在结果末尾用 "
                "<file type=\"image\">路径</file> 附上）。"
            )
        else:
            base = (
                "你是一个专门的子代理(sub-agent)，由主代理委派完成特定子任务。"
                "专注于被委派的任务，完成后直接用纯文本输出最终结果，"
                "不要使用 XML 标签或消息格式（唯一例外：生成的文件/图片可在结果末尾用 "
                "<file type=\"image\">路径</file> 附上，会被真正发送），"
                "不要输出与任务无关的元评论。"
            )
        if not self.inject_framework_brief:
            return base
        tool_names = [t.name for t in tool_set.tools]
        tools_line = ("你可以使用以下工具: " + ", ".join(tool_names)) if tool_names else "你没有可用工具，仅凭自身能力完成。"
        return (
            base
            + tools_line
            + "。工具调用失败时换思路重试，不要重复同样的失败调用。"
            + "若任务无法完全完成，最后一条回复必须说明：已完成的部分、当前进展、卡点。"
        )

    def _apply_llm_steps(self, value, tier: str = "normal") -> int:
        """LLM 提供的步数：应用最小步数地板，再受对应层级硬上限约束。返回 0 表示"未设置"。"""
        try:
            v = int(value or 0)
        except (TypeError, ValueError):
            return 0
        if v <= 0:
            return 0
        if self.min_llm_steps > 0:
            v = max(v, self.min_llm_steps)
        limit = self.coordinator_max_steps_limit if tier == "coordinator" else self.max_steps_limit
        return min(v, limit)

    def _effective_steps(self, cfg: SubAgentConfig) -> int:
        if cfg.tier == "coordinator":
            # 协调者步数独立：默认值与硬上限都不吃全局 max_steps_limit（那个只约束下级）
            return min(cfg.max_steps or self.coordinator_default_steps, self.coordinator_max_steps_limit)
        return min(cfg.max_steps or self.default_max_steps, self.max_steps_limit)

    _TPL_KEYS = ("name", "task_id", "task", "result", "status", "index", "total", "count", "list")

    def _fmt_tpl(self, tpl, **kw) -> str:
        """用户自定义模板容错格式化：未知占位符给空串，格式化失败退回模板原文。"""
        base = {k: "" for k in self._TPL_KEYS}
        base.update({k: ("" if v is None else v) for k, v in kw.items()})
        try:
            # defaultdict：未知占位符替换为空串而不是抛 KeyError 丢掉整条结果
            return str(tpl).format_map(__import__("collections").defaultdict(str, base))
        except Exception:
            return str(tpl)

    def _effective_timeout(self, cfg: SubAgentConfig) -> float:
        if cfg.tier == "coordinator":
            return cfg.timeout if cfg.timeout > 0 else self.coordinator_timeout
        return cfg.timeout if cfg.timeout > 0 else self.default_timeout

    async def _dsml_rescue(self, task: SubAgentTask, cfg: SubAgentConfig, text: str,
                           tool_set: ToolSet, stub_event) -> str:
        """DSML 补执行：provider 把工具调用漏成 DSML 纯文本时，解析出来真的执行，
        并把标记从结果中清除。避免任务"看似完成实际没执行"烂尾。"""
        blocks = _DSML_BLOCK_RE.findall(text)
        if not blocks:
            return text
        cleaned = _DSML_BLOCK_RE.sub("", text).strip()
        appendix: list[str] = []
        for block in blocks:
            for name, body in _DSML_INVOKE_RE.findall(block):
                args = {p: v.strip() for p, v in _DSML_PARAM_RE.findall(body)}
                tool = tool_set.get(name)
                if tool is None or not self._sub_tool_allowed(name, cfg):
                    appendix.append(f"[工具 {name} 不可用或不允许，未补执行]")
                    sub_logger.warning(f"[subagent] DSML 工具 {name} 不可用，未补执行（任务 {task.task_id}）")
                    continue
                try:
                    sub_logger.info(
                        f"[subagent] 检测到 DSML 泄漏，补执行工具 {name}（任务 {task.task_id}）: "
                        f"{json.dumps(args, ensure_ascii=False)[:200]}")
                    if name == "collect_subagent_results":
                        # 与 shim 一致：collect 要等下级跑完，不受单次工具超时约束
                        res = await tool.execute(stub_event, **args)
                    else:
                        res = await asyncio.wait_for(tool.execute(stub_event, **args),
                                                     timeout=self.sub_tool_timeout if self.sub_tool_timeout > 0 else 60)
                    appendix.append(f"[工具 {name} 补执行结果]\n{res}")
                except Exception as e:
                    appendix.append(f"[工具 {name} 补执行失败: {e!r}]")
                    sub_logger.warning(f"[subagent] DSML 补执行失败: {name} {e!r}")
        if appendix:
            return (cleaned + "\n\n" if cleaned else "") + "\n\n".join(appendix)
        return cleaned or text

    async def _execute_subagent(self, task: SubAgentTask, cfg: SubAgentConfig,
                                resume: bool = False, _abandon_retries: int = 0) -> str:
        tool_set = self._build_tool_set(cfg)
        executor = AgentExecutor(
            _SubagentLLMShim(self.ctx.tool_mgr, self.sub_tool_timeout), tool_set)

        if resume and task.request is not None:
            request = task.request
            request.tool_set = tool_set
            request.tools = tool_set.to_list()
            request.tool_choice = "auto" if request.tools else "none"
            request.messages.append(OpenAIMessage(role="user", content=task.task_text))
        else:
            request = LLMRequest(messages=[], tool_set=tool_set)
            if cfg.persona:
                request.system_prompt.append(Prompt(cfg.persona, name="persona", source="system"))
            request.system_prompt.append(Prompt(self._subagent_brief(tool_set, cfg), name="subagent_role", source="system"))
            request.user_prompt.append(Prompt(task.task_text, name="task", source="user"))
            request.assemble_prompt()

        task.request = request
        if self.debug_log_prompts:
            sub_logger.info(
                "[调试] 子代理 %s(%s) 完整请求:\n--- 人设 ---\n%s\n--- 系统简报 ---\n%s\n--- 任务 ---\n%s\n--- 可用工具: %s",
                task.name, task.task_id, cfg.persona or "（无）",
                self._subagent_brief(tool_set, cfg), task.task_text,
                ", ".join(t.name for t in tool_set.tools) or "（无）")
        stub_event = self._make_stub_event(task.sid)
        agent_ctx = AgentExecutionContext(
            event=stub_event, request=request, new_messages=[], model_group=[task.llm_model],
        )

        final_text = ""
        async for step in executor.run(agent_ctx, max_steps=task.max_steps):
            task.current_step = step.step_index
            resp = step.llm_response
            if not resp:
                break
            if resp.text_response:
                final_text = resp.text_response
                task.last_step_summary = "输出: " + resp.text_response.strip()[:80]
                if self.debug_log_steps:
                    sub_logger.info("[调试] 任务 %s 第 %s 步完整输出:\n%s",
                                    task.task_id, step.step_index, resp.text_response)
            if step.has_tool_calls:
                names = [tc.get("function", {}).get("name", "?") for tc in resp.tool_calls]
                task.last_step_summary = f"调用工具: {', '.join(names)}"
                if self.debug_log_steps:
                    sub_logger.info("[调试] 任务 %s 第 %s 步工具调用:\n%s",
                                    task.task_id, step.step_index,
                                    json.dumps(resp.tool_calls, ensure_ascii=False, default=str))
            if step.state == "error":
                raise RuntimeError(step.err or "unknown agent error")
            if not step.has_tool_calls or step.is_final:
                break

        if self.dsml_rescue and final_text and "DSML" in final_text:
            final_text = await self._dsml_rescue(task, cfg, final_text, tool_set, stub_event)

        # 防烂尾规则1：协调者收尾时仍有未交付下级 → 注入提示强制继续（最多追加 3 轮，
        # 全程仍在任务总超时内）。走 resume 路径复用现有请求上下文，注入消息留在历史里
        if cfg.tier == "coordinator" and _abandon_retries < 3:
            pending = self._unfinished_children(task)
            if pending:
                ids = "、".join(f"{t.task_id}({t.name} {t.state})" for t in pending)
                sub_logger.info(
                    f"[subagent] 协调者任务 {task.task_id} 试图收尾但仍有 "
                    f"{len(pending)} 个未交付下级，强制继续")
                original_task_text = task.task_text
                task.task_text = (
                    f"（系统：你还有 {len(pending)} 个下级子代理未交付结果：{ids}。"
                    f"现在不能结束任务。请用 collect_subagent_results 等待并收集它们的结果，"
                    f"或用 stop_subagent 明确放弃不需要的下级，"
                    f"然后汇总所有产出，输出最终完整结果。）"
                )
                try:
                    more = await self._execute_subagent(
                        task, cfg, resume=True, _abandon_retries=_abandon_retries + 1)
                finally:
                    # 恢复用户原始任务文本：注入语只用于续跑这一轮，
                    # 不能污染投递模版 {task}、status 展示和 resume 默认提示
                    task.task_text = original_task_text
                final_text = more or final_text

        if not final_text.strip() and self.force_final_report:
            final_text = await self._wrap_up(task, "步数已用尽但未产出最终结果")
            if self.dsml_rescue and final_text and "DSML" in final_text:
                final_text = await self._dsml_rescue(task, cfg, final_text, tool_set, stub_event)
        return final_text

    async def _rescue_after_wrapup(self, task: SubAgentTask, cfg: SubAgentConfig, text: str) -> str:
        """超时/出错路径的收尾文本也过一遍 DSML 补执行：
        收尾 LLM 即使没有工具也可能漏出 DSML 工具调用标记（服务商问题），
        不补执行就会把标记原样投递给用户（"假完成"）。"""
        if not (self.dsml_rescue and text and "DSML" in text):
            return text
        try:
            tool_set = self._build_tool_set(cfg)
            stub_event = self._make_stub_event(task.sid)
            return await self._dsml_rescue(task, cfg, text, tool_set, stub_event)
        except Exception as e:
            sub_logger.error(f"[subagent] 收尾后 DSML 补执行失败: {e}")
            return text

    async def _wrap_up(self, task: SubAgentTask, reason: str) -> str:
        """强制收尾：不带工具再调用一次，要求必须返回结果或进展说明。"""
        if task.request is None or task.llm_model is None:
            return ""
        try:
            request = task.request
            request.tool_set = ToolSet()
            request.tools = []
            request.tool_choice = "none"
            request.messages.append(OpenAIMessage(
                role="user",
                content=f"（系统：{reason}。这是最后一次回复机会，不能调用任何工具，"
                        f"不要输出任何工具调用标记（DSML 等），必须直接用纯文字输出：已完成的最终结果；"
                        f"若任务未完成，说明已完成部分、当前进展和卡点。"
                        f"如有生成的文件/图片，在末尾用 <file type=\"image\">路径</file> 附上。）",
            ))
            try:
                resp = await asyncio.wait_for(task.llm_model.chat(request), timeout=self.wrapup_timeout)
                return (resp.text_response or "").strip()
            finally:
                # 收尾注入的消息不能留在对话历史里，否则 resume 后 LLM 仍被告知"不能调用工具"
                try:
                    request.messages.pop()
                except Exception:
                    pass
        except Exception as e:
            # e 的 str 可能为空（如部分 TimeoutError/APIError），用 repr 输出完整信息。
            # 完整堆栈仅调试模式输出（平时打出来全是 TLS 底层堆栈，太刷屏）
            if self.debug_enabled:
                import traceback
                sub_logger.warning(f"[subagent] 收尾调用失败: {e!r}\n{traceback.format_exc()}")
            else:
                sub_logger.warning(f"[subagent] 收尾调用失败: {e!r}（完整堆栈见调试模式），将用纯文本请求重试")
            # 兜底：用全新纯文本请求再试一次（避免复用带工具状态的请求体被拒）
            try:
                retry_req = LLMRequest(messages=[], tool_set=ToolSet())
                retry_req.user_prompt.append(Prompt(
                    f"任务：{task.task_text}\n\n（系统：{reason}。不能调用任何工具，"
                    f"不要输出任何工具调用标记（DSML 等），请直接用纯文字输出：已完成的最终结果；"
                    f"若任务未完成，说明已完成部分、当前进展和卡点。"
                    f"如有生成的文件/图片，在末尾用 <file type=\"image\">路径</file> 附上。）",
                    name="wrapup_retry", source="user"))
                retry_req.assemble_prompt()
                resp = await asyncio.wait_for(task.llm_model.chat(retry_req), timeout=self.wrapup_timeout)
                return (resp.text_response or "").strip()
            except Exception as e2:
                import traceback as tb
                if self.debug_enabled:
                    sub_logger.warning(f"[subagent] 收尾重试也失败: {e2!r}\n{tb.format_exc()}")
                else:
                    sub_logger.warning(f"[subagent] 收尾重试也失败: {e2!r}（完整堆栈见调试模式）")
                return ""

    # ------------------------------------------------------------------
    # 任务生命周期
    # ------------------------------------------------------------------

    def _unfinished_children(self, task: SubAgentTask) -> list[SubAgentTask]:
        """协调者尚未交付的下级：在跑的 + 已结束但未被 collect 回收的（stopped 视为已放弃）。"""
        return [t for t in self._tasks.values()
                if t.parent_task_id == task.task_id and not t.collected
                and t.state in ("queued", "running", "done", "timeout", "error")]

    def _purge_coordinator_spawned(self, task: SubAgentTask):
        """协调者任务结束时回收它创建的内存级下级：
        不回收的话 _configs 会随运行时间无限增长，且后续协调者的
        list_subagents 会被一堆过期下级刷屏（浪费 token、容易误派）。
        已保存（持久化）的下级不回收，归属记录也保留——resume 续跑时仍可编辑它们。"""
        owned = self._coordinator_spawned_by.get(task.task_id)
        if not owned:
            return
        for said in list(owned):
            if said in self._coordinator_spawned:  # 只回收内存级
                self._coordinator_spawned.discard(said)
                self._configs.pop(said, None)
                owned.discard(said)
                self._dbg("回收协调者 %s 的内存级下级 %s", task.task_id, said)
        if not owned:
            self._coordinator_spawned_by.pop(task.task_id, None)

    def _cancel_children(self, task: SubAgentTask):
        """防烂尾规则2：协调者结束（完成/停止/超时/出错）时级联取消其仍在跑的下级。"""
        for t in list(self._tasks.values()):
            if t.parent_task_id == task.task_id and t.state in ("queued", "running"):
                self._dbg("级联取消协调者 %s 的下级任务 %s", task.task_id, t.task_id)
                self._stop_task(t)

    def _next_task_id(self) -> str:
        self._task_counter += 1
        return f"t{self._task_counter}"

    def _session_sem(self, sid: str) -> asyncio.Semaphore:
        if sid not in self._session_sems:
            limit = self.max_concurrent_per_session if self.max_concurrent_per_session > 0 else 9999
            self._session_sems[sid] = asyncio.Semaphore(limit)
        return self._session_sems[sid]

    def _spawn(self, cfg: SubAgentConfig, sid: str, origin: str, task_text: str,
               llm_model, steps_override: int = 0, parent_task_id: str = "") -> SubAgentTask:
        task = SubAgentTask(
            task_id=self._next_task_id(),
            subagent_id=cfg.subagent_id,
            name=cfg.name,
            sid=sid,
            origin=origin,
            task_text=task_text,
            created_at=time.time(),
            llm_model=llm_model,
            max_steps=steps_override or self._effective_steps(cfg),
            tier=cfg.tier,
            parent_task_id=parent_task_id,
        )
        task.handle = asyncio.create_task(self._task_runner(task, cfg))
        self._tasks[task.task_id] = task
        self._dbg("派发任务 %s: 子代理=%s(%s), 来源=%s, 会话=%s, 步数上限=%s, 任务=%s",
                  task.task_id, cfg.name, cfg.subagent_id, origin, sid,
                  task.max_steps, task_text[:200])
        return task

    async def _task_runner(self, task: SubAgentTask, cfg: SubAgentConfig, resume: bool = False):
        _ctx_token = _IN_SUBAGENT.set(True)   # 标记子代理上下文，静默框架过程日志
        _task_token = _CURRENT_TASK.set(task)  # 标记当前任务（工具据此识别调用者层级）
        try:
            # 协调者不占并发额度：它大部分时间在 collect 里等下级，
            # 占着信号量会和自己的下级抢（并发=1 时直接死锁到总超时；
            # 多个协调者同跑也会占满名额让下级永远排不上）
            gate = contextlib.AsyncExitStack() if cfg.tier == "coordinator" else self._sem
            sgate = contextlib.AsyncExitStack() if cfg.tier == "coordinator" else self._session_sem(task.sid)
            async with gate:
                async with sgate:
                    task.state = "running"
                    task.started_at = time.time()
                    try:
                        result = await asyncio.wait_for(
                            self._execute_subagent(task, cfg, resume=resume),
                            timeout=self._effective_timeout(cfg),
                        )
                        task.result = result
                        task.state = "done"
                    except asyncio.TimeoutError:
                        sub_logger.warning(f"[subagent] 任务 {task.task_id} 超时")
                        if self.force_final_report:
                            report = await self._wrap_up(task, "执行超时")
                            report = await self._rescue_after_wrapup(task, cfg, report)
                            task.result = report or task.last_step_summary or "（无产出）"
                        else:
                            task.result = task.last_step_summary or "（无产出）"
                        task.state = "timeout"
                    except Exception as e:
                        sub_logger.error(f"[subagent] 任务 {task.task_id} 失败: {e}")
                        task.error = str(e)
                        if self.force_final_report:
                            report = await self._wrap_up(task, f"执行出错：{e}")
                            report = await self._rescue_after_wrapup(task, cfg, report)
                            task.result = report or f"执行出错：{e}"
                        else:
                            task.result = f"执行出错：{e}"
                        task.state = "error"
            task.finished_at = time.time()
            if cfg.tier == "coordinator":
                self._cancel_children(task)   # 防烂尾规则2：级联取消漏网下级
                self._purge_coordinator_spawned(task)  # 回收内存级下级，防 _configs 膨胀
            await self._deliver_result(task)
            self._schedule_task_cleanup(task)
        except asyncio.CancelledError:
            # 手动停止（含排队等信号量期间被取消的情况）
            task.state = "stopped"
            task.finished_at = time.time()
            if cfg.tier == "coordinator":
                self._cancel_children(task)   # 协调者被停止 → 级联取消其下级
                self._purge_coordinator_spawned(task)  # 同步回收内存级下级
            try:
                await self._after_stopped(task)
            except Exception as e:
                sub_logger.error(f"[subagent] 停止后处理失败: {e}")
            self._schedule_task_cleanup(task)
        except Exception as e:
            sub_logger.error(f"[subagent] 任务 {task.task_id} 运行器异常: {e}")
        finally:
            _CURRENT_TASK.reset(_task_token)
            _IN_SUBAGENT.reset(_ctx_token)

    async def _deliver_result(self, task: SubAgentTask):
        """完成投递：命令式且开关关闭 → 直接发用户；否则经缓冲防抖机制通知主 LLM 汇报。"""
        if task.origin == "coordinator":
            # 防烂尾规则3：协调者派出的下级不做正常投递（不通知用户/主 LLM），
            # 结果只经 collect_subagent_results 回流给协调者
            return
        # <file> 里的 data URI 先落盘成真文件（两条投递路径都能发真附件）
        if task.result:
            task.result = _resolve_data_uri_files(task.result, task.task_id)
        if self.strip_msg_tags and task.result:
            task.result = _clean_result_markup(task.result) or task.result
        status_word = {"done": "已完成", "timeout": "因超时结束", "error": "出错结束"}.get(task.state, "已结束")
        if self.debug_log_results:
            sub_logger.info("[调试] 任务 %s(%s) 结束，状态=%s，投递方式=%s，完整结果:\n%s",
                            task.task_id, task.name, task.state,
                            "直接发用户" if (task.origin == "command" and not self.cmd_result_to_main_llm)
                            else "注入缓冲通知主AI",
                            task.result or "（空）")
        if (self.result_delivery == "direct_send"
                or (task.origin == "command" and not self.cmd_result_to_main_llm)):
            try:
                result_text, media = _extract_file_elements(task.result)
                chain = [Text(
                    self._fmt_tpl(self.cmd_result_template,
                        name=task.name, task_id=task.task_id, task=task.task_text,
                        result=result_text, status=status_word,
                    )
                )]
                chain.extend(media)   # <file> 标签 → 真实图片/文件附件
                await self.ctx.send_message_chain(task.sid, MessageChain(chain))
            except Exception as e:
                sub_logger.error(f"[subagent] 结果直发失败: {e}")
            return
        # 合并窗口：稍等片刻，提高与用户消息合并进同一批次的概率
        if self.merge_window_ms > 0:
            await asyncio.sleep(self.merge_window_ms / 1000)
        try:
            await self.ctx.publish_notice(task.sid, MessageChain([Text(
                self._fmt_tpl(self.notify_template,
                    name=task.name, task_id=task.task_id, task=task.task_text,
                    result=task.result, status=status_word,
                )
            )]), is_mentioned=True)
        except Exception as e:
            sub_logger.error(f"[subagent] 结果通知注入失败: {e}")

    async def _after_stopped(self, task: SubAgentTask):
        """手动停止后的处理：保留现场供 resume；按配置返回进度。"""
        if not self.allow_resume:
            task.request = None
        if task.origin == "command" and self.stop_return_progress:
            progress = (
                f"已停止子代理任务 {task.task_id}（{task.name}）。\n"
                f"进度: 第 {task.current_step}/{task.max_steps} 步\n"
                f"最近: {task.last_step_summary or '（暂无）'}"
                + ("\n（可使用继续命令让它接着做）" if self.allow_resume else "")
            )
            try:
                await self.ctx.send_message_chain(task.sid, MessageChain([Text(progress)]))
            except Exception as e:
                sub_logger.error(f"[subagent] 停止进度发送失败: {e}")

    def _schedule_task_cleanup(self, task: SubAgentTask):
        """已结束任务保留一段时间（供查询/resume），过期清理。"""
        if task.state in ("queued", "running"):
            return
        # 协调者现场保留独立时长（二次沟通窗口），普通任务用 resume_keep_minutes
        minutes = self.coordinator_context_minutes if task.tier == "coordinator" else self.resume_keep_minutes
        keep = max(minutes, 1) * 60

        async def _gc():
            await asyncio.sleep(keep)
            cur = self._tasks.get(task.task_id)
            if cur is task and cur.state not in ("queued", "running"):
                self._tasks.pop(task.task_id, None)

        asyncio.create_task(_gc())

    def _stop_task(self, task: SubAgentTask) -> bool:
        if task.state not in ("queued", "running") or not task.handle:
            return False
        self._dbg("停止任务 %s(%s)，停止前状态=%s，进度=%s/%s 步",
                  task.task_id, task.name, task.state, task.current_step, task.max_steps)
        task.handle.cancel()
        # 状态收尾统一由 runner 的 CancelledError 分支完成（即使任务还排在信号量队列里，
        # 协程也已启动并阻塞在 acquire 上，取消后一定会进入 runner 的 except 分支）
        return True

    # ------------------------------------------------------------------
    # LLM 工具：查询 / 注册 / 编辑 / 删除
    # ------------------------------------------------------------------

    def _match_subagents(self, keyword: str) -> list[SubAgentConfig]:
        """按关键词模糊匹配子代理（ID/名称/描述/标签/人设内容），供任务分配参考。"""
        kw = keyword.strip().lower()
        if not kw:
            return list(self._configs.values())
        words = [w for w in re.split(r"[\s,，、;；]+", kw) if w]
        scored: list[tuple[int, SubAgentConfig]] = []
        for cfg in self._configs.values():
            hay_name = f"{cfg.subagent_id} {cfg.name} {cfg.description} {cfg.tags}".lower()
            hay_persona = (cfg.persona or "").lower() if self.keyword_persona_excerpt else ""
            score = 0
            for w in words:
                if w in hay_name:
                    score += 2
                elif hay_persona and w in hay_persona:
                    score += 1
            if score > 0:
                scored.append((score, cfg))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    @register.tool(
        name="list_subagents",
        description="列出当前你可见的子代理(subagent)及其状态、可用模型列表。可传 keyword（任务关键词）模糊匹配最合适的子代理，便于精准分配任务。若已启用协调者：主代理只看到协调者，协调者只看到下级子代理。",
        params={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "可选。任务关键词（可多个，空格分隔），用于匹配最合适的子代理"},
            },
            "required": [],
        },
    )
    async def list_subagents(self, event, keyword: str = "") -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        if not self._configs:
            return "当前没有已注册的子代理。"
        caller = _CURRENT_TASK.get(None)
        if self.enable_coordinator:
            if caller is None:
                # 主 LLM 只见协调者（下级由协调者自行安排）
                configs = [c for c in self._match_subagents(keyword) if c.tier == "coordinator"]
                if not configs and not keyword:
                    return "当前没有可用的协调者，请在插件设置的 coordinator_personas 中配置。"
            else:
                # 协调者只见普通下级（含它自己创建的内存级下级）
                configs = [c for c in self._match_subagents(keyword) if c.tier != "coordinator"]
        else:
            configs = [c for c in self._match_subagents(keyword) if c.tier != "coordinator"]
        if not configs:
            return f"没有匹配「{keyword}」的子代理。请换个关键词，或不传关键词查看全部。"
        lines = []
        if keyword:
            lines.append(f"匹配「{keyword}」的子代理（按相关度排序）:")
        for cfg in configs:
            cid = cfg.subagent_id
            default_mark = ("（协调者）" if cfg.tier == "coordinator"
                            else "（默认）" if cid in self._default_order
                            else "（热加载）" if cid in self._hot_loaded_order else "")
            tier_models = (self.coordinator_models
                           if (cfg.tier == "coordinator" and self.coordinator_models)
                           else self.available_models)
            model_str = cfg.model or ("列表首选" if tier_models else "fast")
            lines.append(
                f"[{cid}] {cfg.name}{default_mark} 来源:{_source_label(cfg.source)} — {cfg.description}\n"
                f"    步数:{self._effective_steps(cfg)}(上限{self.coordinator_max_steps_limit if cfg.tier == 'coordinator' else self.max_steps_limit}) "
                f"超时:{self._effective_timeout(cfg):.0f}s 模型:{model_str} "
                f"工具:{'限定 ' + ','.join(cfg.tools) if cfg.tools else '按插件黑白名单'}"
            )
            if keyword and cfg.persona and self.keyword_persona_excerpt:
                # 关键词模式下附人设摘要（开关控制，默认关以省 token；全文仍受 allow_llm_read_persona 控制）
                limit = max(self.persona_excerpt_length, 10)
                excerpt = re.sub(r"\s+", " ", cfg.persona).strip()[:limit]
                lines.append(f"    人设摘要: {excerpt}{'……' if len(cfg.persona) > limit else ''}")
        running = [t for t in self._tasks.values() if t.state in ("queued", "running")]
        if running:
            lines.append("正在运行的任务: " + ", ".join(
                f"{t.task_id}({t.name} {t.state} {t.current_step}/{t.max_steps}步)" for t in running
            ))
        # 主 LLM 看协调者列表 → 展示协调者模型；协调者看下级列表 → 展示下级模型
        hint_tier = "coordinator" if (caller is None and self.enable_coordinator) else "normal"
        lines.append(f"可选模型: {self._models_hint_text(hint_tier)}")
        header = "" if keyword else "已注册的子代理:\n"
        return header + "\n".join(lines)

    @register.tool(
        name="register_subagent",
        description="动态创建一个新的子代理，之后可通过 spawn_subagent/call_subagent 调用。subagent_id 必须唯一。",
        params={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "子代理唯一标识，例如 'translator'"},
                "name": {"type": "string", "description": "显示名称，例如 '翻译专家'"},
                "description": {"type": "string", "description": "功能描述，供主代理判断何时调用"},
                "persona": {"type": "string", "description": "系统人格设定，描述其角色和能力"},
                "tools": {"type": "array", "items": {"type": "string"},
                          "description": "额外允许的工具名（留空=按插件黑白名单），如 ['sub_read_file','search']"},
                "max_steps": {"type": "integer", "description": "最大推理步数。参考：简单问答/单次工具调用 15 步左右，一般任务 15-20 步，复杂多工具任务（写代码、多文件操作）20-25 步。不确定就填 0 或不填，使用插件默认值（推荐）"},
                "timeout": {"type": "integer", "description": "超时时间（秒）"},
                "model": {"type": "string", "description": "模型，格式 provider_id:model_id 或 fast（须在可选列表内）"},
                "save_persona": {"type": "boolean",
                                 "description": "是否把该子代理保存为 webui 新人设（持久化）。建议：表现有长期复用价值就保存；也可以先不存，之后用 save_subagent_persona 再决定"},
            },
            "required": ["subagent_id", "name", "description", "persona"],
        },
    )
    async def register_subagent_tool(self, event, subagent_id: str, name: str, description: str,
                                     persona: str, tools: list[str] | None = None,
                                     max_steps: int = 0, timeout: int = 0, model: str = "",
                                     save_persona: bool = False) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        caller = _CURRENT_TASK.get(None)
        by_coordinator = caller is not None and caller.tier == "coordinator"
        # 创建总开关对协调者同样生效：用户关掉"允许创建子代理"就是谁都不许建
        if not self.allow_llm_create:
            return "错误: 当前配置不允许创建子代理。"
        if subagent_id in self._configs:
            return f"错误: 子代理 '{subagent_id}' 已存在，请换一个 ID。"
        if model:
            _, err = self._resolve_model(model)
            if err:
                return f"错误: {err}"
        # 保留策略：llm_decide_save 开 → 由主 LLM 通过 save_persona 决定；
        # 关 → 全部自动保存。二者都受总开关 allow_llm_save 约束。
        want_save = save_persona if self.llm_decide_save else True
        if by_coordinator and not self.coordinator_save_spawned:
            want_save = False  # 协调者创建的下级仅内存存在（开关控制），不持久化
        persona_id = ""
        save_note = ""
        if want_save:
            if not self.allow_llm_save:
                if save_persona:
                    return "错误: 当前配置不允许保存子代理为人设（allow_llm_save_persona 已关闭）。"
                save_note = "（未保存为人设：总开关已关闭）"
            else:
                persona_id = await self._save_as_persona(subagent_id, name, persona)
        cfg = SubAgentConfig(
            subagent_id=subagent_id, name=name, description=description, persona=persona,
            persona_id=persona_id, source="coordinator" if by_coordinator else "llm",
            tools=tools or [],
            max_steps=self._apply_llm_steps(max_steps),
            timeout=min(max(float(timeout), 0.0), 3600.0), model=model,  # LLM 传参硬钳制
        )
        self._configs[subagent_id] = cfg
        if by_coordinator:
            # 归属记录（内存级与已保存的都记）：编辑权限按"是不是我这个任务创建的"判定
            self._coordinator_spawned_by.setdefault(caller.task_id, set()).add(subagent_id)
        if by_coordinator and not self.coordinator_save_spawned:
            # 内存级下级：不进 store、不进命令序号，重启即消失
            self._coordinator_spawned.add(subagent_id)
            sub_logger.info(f"Coordinator spawned in-memory sub-agent: {subagent_id} ({name})")
            return f"子代理 '{subagent_id}' ({name}) 注册成功（协调者创建，仅本次运行有效，未持久化）！"
        self._persist_saved(cfg)  # 持久化，重启不丢
        if persona_id and self.hot_reload_saved and subagent_id not in self._hot_loaded_order and subagent_id not in self._default_order:
            self._hot_loaded_order.append(subagent_id)
        sub_logger.info(f"Registered new sub-agent: {subagent_id} ({name}), persona={persona_id or '-'}")
        extra = f"，已保存为人设 {persona_id}" if persona_id else save_note
        return f"子代理 '{subagent_id}' ({name}) 注册成功{extra}！"

    async def _save_as_persona(self, subagent_id: str, name: str, content: str) -> str:
        """把子代理保存为 webui 人设。id 冲突自动加后缀，绝不覆盖已有数据。"""
        persona_mgr = self.ctx.persona_mgr
        pid = f"subagent_{subagent_id}"
        base = pid
        i = 2
        while await persona_mgr.get_persona(pid):
            pid = f"{base}_{i}"
            i += 1
        try:
            existing = await persona_mgr.list_personas()
            existing_names = {p.name for p in existing}
            pname = name
            j = 2
            while pname in existing_names:
                pname = f"{name}-{j}"
                j += 1
            await persona_mgr.create_persona(PersonaInfo(id=pid, name=pname, format="text", content=content))
            return pid
        except Exception as e:
            sub_logger.error(f"[subagent] 保存人设失败: {e}")
            return ""

    @register.tool(
        name="save_subagent_persona",
        description="把一个已存在的子代理保存为 webui 新人设（持久化，重启不丢）。适合子代理任务完成后，判断它有长期复用价值时再保存。绝不覆盖已有人设。",
        params={
            "type": "object",
            "properties": {"subagent_id": {"type": "string", "description": "要保存的子代理 ID"}},
            "required": ["subagent_id"],
        },
    )
    async def save_subagent_persona_tool(self, event, subagent_id: str) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        if not self.allow_llm_save:
            return "错误: 当前配置不允许保存子代理为人设（allow_llm_save_persona 已关闭）。"
        cfg = self._configs.get(subagent_id)
        if not cfg:
            return f"错误: 子代理 '{subagent_id}' 不存在。"
        if cfg.persona_id:
            return f"子代理 '{subagent_id}' 已绑定人设 {cfg.persona_id}，无需重复保存。"
        persona_id = await self._save_as_persona(subagent_id, cfg.name, cfg.persona)
        if not persona_id:
            return "错误: 保存人设失败，详见日志。"
        cfg.persona_id = persona_id
        cfg.source = "llm"
        self._persist_saved(cfg)
        if self.hot_reload_saved and subagent_id not in self._hot_loaded_order and subagent_id not in self._default_order:
            self._hot_loaded_order.append(subagent_id)
        return f"子代理 '{subagent_id}' 已保存为 webui 人设 {persona_id}，可在人设界面编辑。"

    @register.tool(
        name="edit_subagent",
        description="修改已注册子代理的配置。只需提供要修改的字段。内置子代理不可修改（请在 webui 人设界面修改其人设内容）。",
        params={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "要修改的子代理 ID"},
                "name": {"type": "string", "description": "新的显示名称（可选）"},
                "description": {"type": "string", "description": "新的功能描述（可选）"},
                "persona": {"type": "string", "description": "新的系统人格设定（可选）"},
                "tools": {"type": "array", "items": {"type": "string"}, "description": "新的额外工具白名单（可选）"},
                "max_steps": {"type": "integer", "description": "新的最大步数（可选）。参考：简单任务 15 步左右，一般任务 15-20 步，复杂多工具任务 20-25 步"},
                "timeout": {"type": "integer", "description": "新的超时秒数（可选）"},
                "model": {"type": "string", "description": "新的模型（可选），格式 provider_id:model_id 或 fast"},
            },
            "required": ["subagent_id"],
        },
    )
    async def edit_subagent_tool(self, event, subagent_id: str, name: str | None = None,
                                 description: str | None = None, persona: str | None = None,
                                 tools: list[str] | None = None, max_steps: int | None = None,
                                 timeout: int | None = None, model: str | None = None) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        config = self._configs.get(subagent_id)
        if not config:
            return f"错误: 子代理 '{subagent_id}' 不存在。"
        if config.tier == "coordinator":
            # 与 remove 一致：协调者身份/参数只来自 coordinator_personas 配置与 webui 人设，
            # 不允许 LLM（含协调者自己）改，且修改会持久化，影响面大
            return (f"错误: '{subagent_id}' 是协调者，不可通过工具修改"
                    f"（请在插件设置 coordinator_personas 或 webui 人设界面调整）。")
        caller = _CURRENT_TASK.get(None)
        owned = self._coordinator_spawned_by.get(caller.task_id, set()) if caller is not None else set()
        if caller is not None and caller.tier == "coordinator" and subagent_id not in owned:
            # 协调者只能改【自己这个任务】创建的下级（含内存级与已保存的）；
            # 内置/用户人设/主AI创建的、以及【其他协调者】创建的都不归它管
            return (f"错误: 你只能修改自己创建的下级子代理，"
                    f"'{subagent_id}' 是{_source_label(config.source)}子代理，不归你管。")
        if config.source == "builtin" and persona is not None:
            return f"错误: '{subagent_id}' 是内置子代理，请在 webui 人设界面修改其人设内容。"
        if model:
            _, err = self._resolve_model(model, config.tier)
            if err:
                return f"错误: {err}"
        if name is not None:
            config.name = name
        if description is not None:
            config.description = description
        if persona is not None:
            config.persona = persona
        if tools is not None:
            config.tools = tools
        if max_steps is not None:
            config.max_steps = self._apply_llm_steps(max_steps, config.tier)
        if timeout is not None:
            config.timeout = min(max(float(timeout), 0.0), 3600.0)  # LLM 传参硬钳制
        if model is not None:
            config.model = model
        # 所有来源都持久化：内置/用户人设子代理以 override 形式存（人设文本仍以 webui 为准），
        # 步数/超时/模型等修改重启不丢；内存级下级（协调者创建未保存的）除外
        if subagent_id in self._coordinator_spawned:
            self._dbg("内存级下级 %s 的修改仅本次有效，不持久化", subagent_id)
        else:
            self._persist_saved(config)
        sub_logger.info(f"Edited sub-agent: {subagent_id}")
        return (f"子代理 '{subagent_id}' 已更新。当前: 名称={config.name}, 步数={config.max_steps or self.default_max_steps}, "
                f"超时={self._effective_timeout(config):.0f}s, 模型={config.model or '默认'}, 工具={config.tools or '按黑白名单'}")

    @register.tool(
        name="remove_subagent",
        description="删除一个已注册的子代理。内置子代理不可删除。正在运行的同名任务不受影响。",
        params={
            "type": "object",
            "properties": {"subagent_id": {"type": "string", "description": "要删除的子代理 ID"}},
            "required": ["subagent_id"],
        },
    )
    async def remove_subagent_tool(self, event, subagent_id: str) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        cfg = self._configs.get(subagent_id)
        if not cfg:
            return f"错误: 子代理 '{subagent_id}' 不存在。"
        if cfg.tier == "coordinator":
            return f"错误: '{subagent_id}' 是协调者，不可删除（可在插件设置 coordinator_personas 中移除）。"
        if cfg.source == "builtin":
            return f"错误: '{subagent_id}' 是内置子代理，不可删除（可在插件设置 default_personas 中移除）。"
        del self._configs[subagent_id]
        if subagent_id in self._default_order:
            self._default_order.remove(subagent_id)
        if subagent_id in self._hot_loaded_order:
            self._hot_loaded_order.remove(subagent_id)
        self._store["saved"].pop(subagent_id, None)
        self._store_save()
        sub_logger.info(f"Removed sub-agent: {subagent_id}")
        return f"子代理 '{subagent_id}' 已删除。"

    @register.tool(
        name="get_subagent_persona",
        description="查看某个子代理的完整人设内容与参数。仅在确实需要了解其详细设定时调用。",
        params={
            "type": "object",
            "properties": {"subagent_id": {"type": "string", "description": "子代理 ID"}},
            "required": ["subagent_id"],
        },
    )
    async def get_subagent_persona_tool(self, event, subagent_id: str) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        if not self.allow_llm_read_persona:
            return "当前配置不允许查看子代理人设详情。"
        cfg = self._configs.get(subagent_id)
        if not cfg:
            return f"错误: 子代理 '{subagent_id}' 不存在。"
        # 人设内容以 webui 实时内容为准
        persona_text = cfg.persona
        if cfg.persona_id:
            p = await self.ctx.persona_mgr.get_persona(cfg.persona_id)
            if p:
                persona_text = p.content or persona_text
        return (f"[{subagent_id}] {cfg.name}（来源:{_source_label(cfg.source)}，人设ID:{cfg.persona_id or '无'}）\n"
                f"描述: {cfg.description}\n步数: {cfg.max_steps or self.default_max_steps} "
                f"超时: {self._effective_timeout(cfg):.0f}s 模型: {cfg.model or '默认'}\n"
                f"人设内容:\n{persona_text}")

    # ------------------------------------------------------------------
    # LLM 工具：派发 / 同步调用 / 状态 / 停止 / 继续
    # ------------------------------------------------------------------

    @register.tool(
        name="spawn_subagent",
        description="异步派出一个子代理执行子任务：立即返回任务ID，子代理在后台运行，完成后系统会通知你结果（你再向用户汇报）。适合耗时或需专业能力的子任务。可用 list_subagents 查看可用子代理。若已启用协调者：你只能派出协调者，由它拆解任务；协调者用本工具可一步内连派多个下级并行执行，再用 collect_subagent_results 收集结果。",
        params={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "子代理ID"},
                "task": {"type": "string", "description": "需要完成的具体任务描述"},
                "model": {"type": "string", "description": "可选，本次使用的模型（须在可选列表内），格式 provider_id:model_id 或 fast"},
                "max_steps": {"type": "integer", "description": "可选，本次任务的步数覆盖（不改子代理配置）。参考：简单问答/单次工具调用 15 步左右，一般任务 15-20 步，复杂多工具任务 20-25 步。不确定就填 0 或不填"},
            },
            "required": ["subagent_id", "task"],
        },
    )
    async def spawn_subagent(self, event, subagent_id: str, task: str, model: str = "",
                             max_steps: int = 0) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        cfg = self._configs.get(subagent_id)
        if not cfg:
            return f"错误: 子代理 '{subagent_id}' 不存在。可用 list_subagents 查看。"
        # 两层级管控：主 LLM（无任务上下文）只能派协调者；协调者只能派普通下级；
        # 普通子代理根本拿不到本工具（_SAFETY_BOTTOM），此处防御性兜底
        caller = _CURRENT_TASK.get(None)
        origin, parent_task_id = "tool", ""
        if self.enable_coordinator:
            if caller is None:
                if cfg.tier != "coordinator":
                    coords = "、".join(
                        f"{c.name}[{c.subagent_id}]" for c in self._configs.values()
                        if c.tier == "coordinator") or "（无）"
                    return (f"错误: 已启用协调者，你只能把任务交给协调者，由它拆解并安排下级子代理。"
                            f"可用协调者: {coords}")
            elif caller.tier == "coordinator":
                if cfg.tier == "coordinator":
                    return "错误: 协调者不能再派出协调者，请派给下级子代理。"
                origin, parent_task_id = "coordinator", caller.task_id
            else:
                return "错误: 普通子代理不能再派出子代理。"
        elif cfg.tier == "coordinator":
            return "错误: 协调者功能未启用（请在插件设置中打开 enable_coordinator）。"
        llm_model, err = self._resolve_model(model or cfg.model, cfg.tier)
        if err:
            return f"错误: {err}"
        t = self._spawn(cfg, event.sid, origin, task, llm_model,
                        steps_override=self._apply_llm_steps(max_steps, cfg.tier),
                        parent_task_id=parent_task_id)
        sub_logger.info(f"Spawned task {t.task_id} ({subagent_id}) in {event.sid}")
        if origin == "coordinator":
            return (f"已派出下级子代理「{cfg.name}」（任务ID {t.task_id}，最多 {t.max_steps} 步）。"
                    f"它会后台并行执行，不会打扰主代理；用 collect_subagent_results 等待并收集结果，"
                    f"期间可用 subagent_status 查询进度，或用 stop_subagent 放弃。")
        return (f"已派出子代理「{cfg.name}」（任务ID {t.task_id}，最多 {t.max_steps} 步）。"
                f"它会后台执行，完成后你会收到结果通知；期间可用 subagent_status 查询进度，"
                f"或用 stop_subagent 停止。")

    @register.tool(
        name="collect_subagent_results",
        description="（仅协调者可用）等待你派出的所有未收集下级子代理运行结束，把它们的产出一次性返回给你。派出下级后务必用它收集结果再汇总交付；可传 task_ids 只收集指定下级。",
        params={
            "type": "object",
            "properties": {
                "task_ids": {"type": "array", "items": {"type": "string"},
                             "description": "可选，只收集这些任务ID；留空收集全部未收集的下级"},
            },
            "required": [],
        },
    )
    async def collect_subagent_results(self, event, task_ids: list[str] | None = None) -> str:
        caller = _CURRENT_TASK.get(None)
        if caller is None or caller.tier != "coordinator":
            return "错误: 该工具仅供协调者使用（主代理派出任务后等系统通知即可，无需收集）。"
        children = [t for t in self._tasks.values()
                    if t.parent_task_id == caller.task_id and not t.collected]
        if task_ids:
            wanted = set(task_ids)
            children = [t for t in children if t.task_id in wanted]
        if not children:
            return "当前没有未收集的下级任务（可能都已收集、被放弃或尚未派出）。"
        # 等所有目标下级结束（含排队中的）。不设工具级超时：
        # 上限由协调者任务总超时（coordinator_timeout）兜底
        handles = [t.handle for t in children if t.handle and not t.handle.done()]
        if handles:
            await asyncio.gather(*handles, return_exceptions=True)
        parts = []
        for t in children:
            if t.collected:
                continue
            t.collected = True
            body = t.result or t.error or t.last_step_summary or "（无产出）"
            parts.append(f"── 下级 {t.task_id}（{t.name}，状态 {t.state}）──\n{body}")
        return "已收集 {} 个下级结果：\n\n{}".format(len(parts), "\n\n".join(parts))

    @register.tool(
        name="call_subagent",
        description="同步调用子代理：阻塞等待其完成并直接返回结果。注意：调用期间你无法回复用户，耗时任务请改用 spawn_subagent。",
        params={
            "type": "object",
            "properties": {
                "subagent_id": {"type": "string", "description": "子代理ID"},
                "task": {"type": "string", "description": "需要完成的具体任务描述"},
                "model": {"type": "string", "description": "可选，本次使用的模型（须在可选列表内）"},
            },
            "required": ["subagent_id", "task"],
        },
    )
    async def call_subagent(self, event, subagent_id: str, task: str, model: str = "") -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        cfg = self._configs.get(subagent_id)
        if not cfg:
            return f"Error: SubAgent '{subagent_id}' not found. Available: {list(self._configs.keys())}"
        if self.enable_coordinator and cfg.tier != "coordinator":
            return "错误: 已启用协调者，请改用 spawn_subagent 把任务交给协调者，由它拆解并安排下级子代理。"
        if cfg.tier == "coordinator" and not self.enable_coordinator:
            return "错误: 协调者功能未启用（请在插件设置中打开 enable_coordinator）。"
        if self.call_mode == "async":
            # 异步模式：同步入口转为派发，避免阻塞主循环
            return await self.spawn_subagent(event, subagent_id, task, model)
        llm_model, err = self._resolve_model(model or cfg.model, cfg.tier)
        if err:
            return f"Error: {err}"
        t = SubAgentTask(
            task_id=self._next_task_id(), subagent_id=cfg.subagent_id, name=cfg.name,
            sid=event.sid, origin="tool", task_text=task, created_at=time.time(),
            llm_model=llm_model, max_steps=self._effective_steps(cfg), tier=cfg.tier,
        )
        self._tasks[t.task_id] = t
        t.state = "running"
        t.started_at = time.time()
        # 与异步路径一致：同步执行也要标记当前任务，否则协调者会被当成主 LLM
        # （派下级被拒/注册走错持久化路径），且结束时不会级联清理下级
        _task_token = _CURRENT_TASK.set(t)
        try:
            result = await asyncio.wait_for(
                self._execute_subagent(t, cfg), timeout=self._effective_timeout(cfg)
            )
            t.state = "done"
            t.result = result
        except asyncio.TimeoutError:
            t.state = "timeout"
            report = await self._wrap_up(t, "执行超时") if self.force_final_report else ""
            return f"SubAgent '{subagent_id}' 超时结束。{('进展：' + report) if report else ''}"
        except Exception as e:
            t.state = "error"
            sub_logger.error(f"SubAgent '{subagent_id}' error: {e}")
            return f"Error: SubAgent '{subagent_id}' failed: {e}"
        finally:
            t.finished_at = time.time()
            if cfg.tier == "coordinator":
                self._cancel_children(t)            # 同步路径同样级联取消漏网下级
                self._purge_coordinator_spawned(t)  # 同步回收内存级下级
            _CURRENT_TASK.reset(_task_token)
            self._schedule_task_cleanup(t)
        return f"SubAgent '{cfg.name}' result:\n{t.result}"

    @register.tool(
        name="subagent_status",
        description="查看子代理任务进度。不传 task_id 时列出所有任务的紧凑状态；传 task_id 查看该任务详情（当前步数、最近一步情况、结果等）。",
        params={
            "type": "object",
            "properties": {"task_id": {"type": "string", "description": "任务ID，如 t1。留空列出全部"}},
            "required": [],
        },
    )
    async def subagent_status(self, event, task_id: str = "") -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        # 防呆循环：限制最小连续查询间隔。
        # 协调者与主 LLM 共用会话 sid，按调用者分别限流，互不挤占额度
        caller = _CURRENT_TASK.get(None)
        rl_key = f"{event.sid}#{caller.task_id}" if caller is not None else event.sid
        now = time.time()
        last = self._last_status_query.get(rl_key, 0.0)
        if self.status_query_interval > 0 and now - last < self.status_query_interval:
            remain = int(self.status_query_interval - (now - last))
            active = [t for t in self._tasks.values() if t.state in ("queued", "running")]
            brief = ("、".join(f"{t.task_id}({t.name} {t.state} {t.current_step}/{t.max_steps}步)" for t in active)
                     or "暂无运行中任务")
            return (f"查询过于频繁：最小查询间隔为 {int(self.status_query_interval)} 秒，"
                    f"请约 {remain} 秒后再查。任务完成后会主动通知你，无需反复查询。\n"
                    f"当前概况: {brief}")
        self._last_status_query[rl_key] = now
        if not task_id:
            if not self._tasks:
                return "当前没有任何子代理任务记录。"
            lines = []
            caller = _CURRENT_TASK.get(None)
            for t in self._tasks.values():
                if t.sid != event.sid:
                    continue  # 会话隔离：不暴露其他会话的任务
                if caller is not None and caller.tier == "coordinator" and t.parent_task_id != caller.task_id:
                    continue  # 协调者只能看到自己派出的下级
                elapsed = int((t.finished_at or time.time()) - t.created_at)
                lines.append(
                    f"[{t.task_id}] {t.name}({t.subagent_id}) {t.state} "
                    f"{t.current_step}/{t.max_steps}步 {elapsed}s | 最近: {t.last_step_summary[:60] or '—'}"
                )
            return "子代理任务:\n" + "\n".join(lines)
        t = self._tasks.get(task_id)
        if not t or t.sid != event.sid:
            return f"错误: 任务 '{task_id}' 不存在或已被清理。"
        caller = _CURRENT_TASK.get(None)
        if caller is not None and caller.tier == "coordinator" and t.parent_task_id != caller.task_id:
            return f"错误: 任务 '{task_id}' 不是你派出的下级。"
        detail = (
            f"任务 {t.task_id}: 子代理 {t.name}({t.subagent_id})\n"
            f"状态: {t.state} | 步数: {t.current_step}/{t.max_steps} | 来源: {t.origin}\n"
            f"任务: {t.task_text[:200]}\n"
            f"最近一步: {t.last_step_summary or '—'}\n"
        )
        if t.error:
            detail += f"错误: {t.error}\n"
        if t.result and t.state in ("done", "timeout", "error"):
            detail += f"结果:\n{t.result}"
        elif t.state == "stopped":
            detail += "已停止，可用 resume_subagent 继续。" if self.allow_resume else "已停止。"
        return detail

    @register.tool(
        name="stop_subagent",
        description="停止一个正在运行的子代理任务。任务现场会保留，之后可用 resume_subagent 让它接着做。",
        params={
            "type": "object",
            "properties": {"task_id": {"type": "string", "description": "要停止的任务ID"}},
            "required": ["task_id"],
        },
    )
    async def stop_subagent(self, event, task_id: str) -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        t = self._tasks.get(task_id)
        if not t or t.sid != event.sid:
            return f"错误: 任务 '{task_id}' 不存在。"
        caller = _CURRENT_TASK.get(None)
        if caller is not None and caller.tier == "coordinator":
            if t.parent_task_id != caller.task_id:
                return f"错误: 任务 '{task_id}' 不是你派出的下级。"
        if not self._stop_task(t):
            return f"任务 '{task_id}' 当前状态为 {t.state}，无法停止。"
        if caller is not None and caller.tier == "coordinator":
            t.collected = True  # 明确放弃：不再阻塞协调者收尾
        progress = f"已停止任务 {task_id}（{t.name}），停在第 {t.current_step}/{t.max_steps} 步。最近: {t.last_step_summary or '—'}"
        if self.allow_resume:
            progress += "。可用 resume_subagent 让它继续。"
        return progress

    @register.tool(
        name="resume_subagent",
        description="让一个已停止/超时/出错的子代理任务接着之前的进度继续执行，可追加新的指示。",
        params={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "要继续的任务ID"},
                "instruction": {"type": "string", "description": "追加指示（可选），如 '接着写第二章'"},
            },
            "required": ["task_id"],
        },
    )
    async def resume_subagent(self, event, task_id: str, instruction: str = "") -> str:
        if not self._in_scope(event.sid):
            return "当前会话未启用子代理功能。"
        if not self.allow_resume:
            return "当前配置不允许恢复子代理任务。"
        t = self._tasks.get(task_id)
        if not t or t.sid != event.sid:
            return f"错误: 任务 '{task_id}' 不存在或已被清理。"
        caller = _CURRENT_TASK.get(None)
        if caller is not None and caller.tier == "coordinator" and t.parent_task_id != caller.task_id:
            return f"错误: 任务 '{task_id}' 不是你派出的下级。"
        if t.state in ("queued", "running"):
            return f"任务 '{task_id}' 正在运行中，无需恢复。"
        if t.request is None or t.llm_model is None:
            return f"错误: 任务 '{task_id}' 的执行现场已不可用，无法恢复。"
        cfg = self._configs.get(t.subagent_id)
        if not cfg:
            return f"错误: 子代理 '{t.subagent_id}' 已被删除。"
        t.task_text = instruction or "请接着之前的进度继续完成任务。"
        t.state = "queued"
        t.current_step = 0
        t.max_steps = self._effective_steps(cfg)
        t.error = ""
        t.handle = asyncio.create_task(self._task_runner(t, cfg, resume=True))
        return f"已恢复任务 {task_id}（{t.name}），将继续执行最多 {t.max_steps} 步，完成后会通知你结果。"

    # ------------------------------------------------------------------
    # 命令系统（不进 LLM 上下文；命令 + 工具双通道）
    # ------------------------------------------------------------------

    def _extract_text(self, event: KiraMessageEvent) -> str:
        return "".join(e.text for e in event.message.chain if isinstance(e, Text)).strip()

    async def _cmd_reply(self, event: KiraMessageEvent, text: str):
        try:
            await self.ctx.send_message_chain(event.session.sid, MessageChain([Text(text)]))
        except Exception as e:
            sub_logger.error(f"[subagent] 命令回复发送失败: {e}")
        event.discard(force=True)
        event.stop()

    @on.im_message(priority=Priority.HIGH)
    async def handle_commands(self, event: KiraMessageEvent):
        if not self.enable_commands:
            return
        text = self._extract_text(event)
        if not text:
            return
        sid = event.session.sid
        if not self._in_scope(sid):
            return

        # 命令白名单（仅约束命令通道）：命中命令且名单非空时校验 QQ 号
        if self.cmd_allowed_users:
            is_stop = any(text == a or text.startswith(a + " ") for a in self.cmd_stop_aliases)
            is_resume = any(text == a or text.startswith(a + " ") for a in self.cmd_resume_aliases)
            is_start = any(re.match(rf"^{re.escape(a)}\d+", text) for a in self.cmd_start_aliases)
            if is_stop or is_resume or is_start:
                user_id = self._event_user_id(event)
                if not user_id or user_id not in self.cmd_allowed_users:
                    await self._cmd_reply(event, self.cmd_denied_message)
                    return

        # 停止命令：/stopsuba [任务ID|序号]
        for alias in self.cmd_stop_aliases:
            if text == alias or text.startswith(alias + " "):
                await self._cmd_stop(event, text[len(alias):].strip())
                return

        # 继续命令：/resumesuba <任务ID|序号> [新指示]（序号可紧贴命令：/resumesuba1 xxx）
        for alias in self.cmd_resume_aliases:
            if text == alias or text.startswith(alias + " "):
                await self._cmd_resume(event, text[len(alias):].strip())
                return
            m = re.match(rf"^{re.escape(alias)}(\d+)(?:\s+(.*))?$", text, re.S)
            if m:
                rest = m.group(1) + (" " + m.group(2) if m.group(2) else "")
                await self._cmd_resume(event, rest)
                return

        # 启动命令：/suba1 <任务内容>（序号对应默认子代理列表顺序）
        for alias in self.cmd_start_aliases:
            m = re.match(rf"^{re.escape(alias)}(\d+)\s*(.*)$", text, re.S)
            if m:
                await self._cmd_start(event, int(m.group(1)), m.group(2).strip())
                return

    @staticmethod
    def _event_user_id(event: KiraMessageEvent) -> str:
        """从事件中取发送者 QQ 号（取不到返回空串）。"""
        sender = getattr(getattr(event, "message", None), "sender", None)
        uid = getattr(sender, "user_id", None)
        return str(uid) if uid is not None else ""

    def _cmd_order(self) -> list[str]:
        """命令序号对应的完整子代理顺序。
        协调者开启时：序号直接对应协调者列表（普通下级不再接受命令直派）；
        关闭时：默认列表 + （开关允许时）热加载的 AI 子代理。"""
        if self.enable_coordinator:
            return [sid_ for sid_ in self._coordinator_order if sid_ in self._configs]
        order = [sid_ for sid_ in self._default_order if sid_ in self._configs]
        if self.cmd_include_hot_loaded:
            order += [sid_ for sid_ in self._hot_loaded_order
                      if sid_ in self._configs and sid_ not in order]
        return order

    def _default_list_text(self) -> str:
        order = self._cmd_order()
        if not order:
            return "（空）"
        return "\n".join(
            f"{i + 1}. {self._configs[sid_].name} [{sid_}]（{_source_label(self._configs[sid_].source)}）"
            for i, sid_ in enumerate(order)
        )

    async def _cmd_start(self, event: KiraMessageEvent, index: int, task_text: str):
        order = self._cmd_order()
        if not order:
            await self._cmd_reply(event, self.msg_no_default)
            return
        if index < 1 or index > len(order):
            await self._cmd_reply(event, self._fmt_tpl(self.msg_invalid_index,
                index=index, list=self._default_list_text()))
            return
        cfg = self._configs.get(order[index - 1])
        if not cfg:
            await self._cmd_reply(event, self._fmt_tpl(self.msg_invalid_index,
                index=index, list=self._default_list_text()))
            return
        if not task_text:
            await self._cmd_reply(event, f"请在命令后加上任务内容，例如：{self.cmd_start_aliases[0]}{index} 帮我审查这段代码")
            return
        llm_model, err = self._resolve_model(cfg.model, cfg.tier)
        if err:
            await self._cmd_reply(event, f"启动失败：{err}")
            return
        t = self._spawn(cfg, event.session.sid, "command", task_text, llm_model)
        sub_logger.info(f"Command spawned task {t.task_id} ({cfg.subagent_id}) in {event.session.sid}")
        await self._cmd_reply(event, self._fmt_tpl(self.msg_started,
            name=cfg.name, task_id=t.task_id, task=task_text, index=index))

    async def _cmd_stop(self, event: KiraMessageEvent, arg: str):
        sid = event.session.sid
        active = [t for t in self._tasks.values()
                  if t.sid == sid and t.state in ("queued", "running")]
        if not active:
            await self._cmd_reply(event, self.msg_none_running)
            return
        stopped = []
        if not arg:
            targets = active  # 停止本会话全部
        else:
            t = self._tasks.get(arg)
            if t and t.sid == sid and t.state in ("queued", "running"):
                targets = [t]
            elif arg.isdigit() and 1 <= int(arg) <= len(active):
                targets = [active[int(arg) - 1]]  # 按运行中任务的序号
            else:
                await self._cmd_reply(event, f"找不到对应的运行中任务：{arg}\n" + "\n".join(
                    f"{i + 1}. [{t.task_id}] {t.name} {t.state} {t.current_step}/{t.max_steps}步"
                    for i, t in enumerate(active)))
                return
        for t in targets:
            if self._stop_task(t):
                stopped.append(t)
        if not stopped:
            await self._cmd_reply(event, self.msg_none_running)
            return
        names = "、".join(f"{t.task_id}({t.name})" for t in stopped)
        # 进度详情由 _after_stopped 按 stop_return_progress 配置逐任务发送
        if not self.stop_return_progress:
            await self._cmd_reply(event, self._fmt_tpl(self.msg_stopped, task_id=names, name=""))
        else:
            event.discard(force=True)
            event.stop()

    async def _cmd_resume(self, event: KiraMessageEvent, arg: str):
        """/resumesuba <任务ID|序号> [新指示]
        序号 = 列表序号（协调者开启时对应协调者列表，否则对应普通子代理列表），
        自动找该子代理在本会话最近一个已结束且现场未过期的任务继续。"""
        if not self.allow_resume:
            await self._cmd_reply(event, "当前配置不允许恢复子代理任务。")
            return
        parts = arg.split(maxsplit=1)
        target = parts[0] if parts else ""
        instruction = parts[1] if len(parts) > 1 else ""
        if target.isdigit():
            # 序号模式：找该序号对应子代理的最近可继续任务
            order = self._cmd_order()
            idx = int(target)
            if not order:
                await self._cmd_reply(event, self.msg_no_default)
                return
            if not (1 <= idx <= len(order)):
                await self._cmd_reply(event, self._fmt_tpl(self.msg_invalid_index,
                    index=idx, list=self._default_list_text()))
                return
            said = order[idx - 1]
            candidates = [x for x in self._tasks.values()
                          if x.sid == event.session.sid and x.subagent_id == said
                          and x.state not in ("queued", "running")]
            candidates.sort(key=lambda x: x.finished_at or 0)
            if not candidates:
                await self._cmd_reply(
                    event,
                    f"「{self._configs[said].name}」当前没有可继续的任务。"
                    f"（只有本会话内已结束、且现场还在保留期内的任务才能继续）")
                return
            t = candidates[-1]
        else:
            t = self._tasks.get(target)
            if not t or t.sid != event.session.sid:
                await self._cmd_reply(event, f"找不到任务 {target or '（未指定）'}。")
                return
        if t.state in ("queued", "running"):
            await self._cmd_reply(event, f"任务 {t.task_id} 正在运行中。")
            return
        if t.request is None or t.llm_model is None:
            await self._cmd_reply(event, f"任务 {t.task_id} 的执行现场已不可用，无法恢复。")
            return
        cfg = self._configs.get(t.subagent_id)
        if not cfg:
            await self._cmd_reply(event, f"子代理 {t.subagent_id} 已被删除，无法恢复任务。")
            return
        t.task_text = instruction or "请接着之前的进度继续完成任务。"
        t.state = "queued"
        t.current_step = 0
        t.max_steps = self._effective_steps(cfg)
        t.error = ""
        t.handle = asyncio.create_task(self._task_runner(t, cfg, resume=True))
        await self._cmd_reply(event, f"已让子代理「{t.name}」（任务 {t.task_id}）继续工作，完成后我会告诉你。")
