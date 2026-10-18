"""Regression checks for config improvements independent of selection order."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def plugin_module():
    try:
        # KiraAI 3.0 split the registry out of core.plugin; 2.x keeps the
        # module at core.plugin.plugin_registry with the same attributes.
        from core.plugin import registry
    except ImportError:
        from core.plugin import plugin_registry as registry

    module_name = "subagent_configuration_test"
    spec = importlib.util.spec_from_file_location(module_name, PLUGIN_ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    registry._module_to_plugin[module_name] = module_name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        registry._plugin_components.pop(module_name, None)
        registry._module_to_plugin.pop(module_name, None)
        sys.modules.pop(module_name, None)


def schema_fields():
    from core.config.config_field import build_fields

    schema = json.loads((PLUGIN_ROOT / "schema.json").read_text(encoding="utf-8"))
    return {field.key: field.to_dict() for field in build_fields(schema)}


def test_session_picker_keeps_existing_list_values_and_locales():
    field = schema_fields()["section_general"]["fields"]["enabled_sessions"]
    assert field["type"] == "multi_select"
    assert field["source"] == "session"
    assert field["allow_custom"] is True
    assert field["default"] == []
    assert field["name"] and field["hint"]
    assert field["locales"]["zh"]["name"] and field["locales"]["zh"]["hint"]


def test_session_permissions_do_not_depend_on_display_order(plugin_module):
    sessions = ["qq:gm:123", "telegram:dm:456"]
    for selected in (sessions, list(reversed(sessions)), []):
        cfg = {"section_general": {"enabled_sessions": copy.deepcopy(selected)}}
        plugin = plugin_module.SubAgentPlugin(None, cfg)
        plugin._load_cfg()
        assert plugin.enabled_sessions == selected
        assert plugin._in_scope("qq:gm:123")
        assert plugin._in_scope("telegram:dm:456")
        assert plugin._in_scope("qq:dm:789") is (not selected)
        assert cfg["section_general"]["enabled_sessions"] == selected


def test_mode_dropdowns_keep_defaults_and_choices():
    fields = schema_fields()["section_general"]["fields"]
    for key, default, options in (
        ("call_mode", "async", ["async", "sync"]),
        ("result_delivery", "to_main_llm", ["to_main_llm", "direct_send"]),
    ):
        assert fields[key]["type"] == "string"
        assert fields[key]["default"] == default
        assert fields[key]["options"] == options


def test_order_sensitive_settings_keep_the_original_inline_format(plugin_module):
    sections = schema_fields()
    for section, key in (
        ("section_models", "available_models"),
        ("section_persona", "default_personas"),
        ("section_coordinator", "coordinator_personas"),
        ("section_coordinator", "coordinator_models"),
    ):
        field = sections[section]["fields"][key]
        assert field["type"] == "list"
        assert "source" not in field
    cfg = {"section_models": {"available_models": ["p;first;coding", "p;second;writing"]}}
    plugin = plugin_module.SubAgentPlugin(None, cfg)
    plugin._load_cfg()
    assert [model["hint"] for model in plugin.available_models] == ["coding", "writing"]
    assert plugin.default_personas_raw == ["子代理-代码专家;代码 审查 重构 调试", "子代理-写作专家;写作 润色 文案"]
