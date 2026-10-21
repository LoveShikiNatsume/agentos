from pathlib import Path

import agentos.dashboard

STATIC = Path(agentos.dashboard.__file__).parent / "static"


def _read(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


def test_director_results_use_evidence_zones_not_table_rows():
    js = _read("director.js")

    for marker in [
        "dir-results-overview",
        "dir-evidence-tracks",
        "dir-recovery-proof",
        "renderOverview",
        "renderEvidenceTracks",
        "renderRecoveryProof",
        "dir-card-closing",
    ]:
        assert marker in js

    assert "dir-data-foot" not in js
    assert "dir-data-badges" not in js
    assert "dir-data-rows" not in js


def test_director_css_has_qhd_evidence_layout_and_no_footnote_style():
    css = _read("director.css")

    for marker in [
        "#dir-results-overview",
        ".dir-evidence-track",
        "#dir-recovery-proof",
        ".dir-source-pill",
        "#dir-card-closing",
        "@media (min-width: 2200px)",
        "clamp(",
    ]:
        assert marker in css

    assert "#dir-data-foot" not in css
    assert "#dir-data-badges" not in css
    assert "#dir-data-rows" not in css


def test_index_has_two_dashboard_pages_and_zone_dom_ids():
    html = _read("index.html")
    for dom_id in [
        "page-local", "page-cloud", "tab-local", "tab-cloud",
        "zone-process", "zone-memory", "zone-cache",
        "curve", "ref-tbody", "deny-tbody", "fault-tl",
        "cor-tbody", "sub-tbody", "cache-tbody", "dag-view",
    ]:
        assert f'id="{dom_id}"' in html, f"missing DOM id: {dom_id}"


def test_index_links_external_css_and_js():
    html = _read("index.html")
    assert '<link rel="stylesheet" href="/style.css">' in html
    assert '<script src="/app.js"></script>' in html
    assert "<style>" not in html  # styles must live in style.css
    assert 'AgentOS' in html


def test_style_has_warm_keynote_palette():
    css = _read("style.css")
    assert "#14b8a6" in css   # accent / cache (teal)
    assert "#eef5fc" in css   # shell (competition pale blue)
    assert "#0f1f3a" in css   # ink (navy)
    assert "--proc" in css and "--mem" in css and "--cache" in css


def test_local_page_highlights_four_core_modules():
    html = _read("index.html")
    for label in [
        "1 进程 / Agent 调度",
        "2 DAG 调度流",
        "3 上下文内存",
        "4 缓存亲和",
    ]:
        assert label in html
    assert "复杂多 Agent 场景验证" in html
    assert "国产开源 OS 支持" in html
    assert "评分映射" in html


def test_local_page_has_runtime_baseline_comparison_tags():
    html = _read("index.html")
    for label in [
        "vs Baseline: makespan -31%",
        "wait -42%",
        "vs Baseline: fault recovered 8/8",
        "baseline halted",
        "vs Baseline: prompt tokens -38%",
        "duplicate context -52%",
        "vs Baseline: KV hit +4.7pp",
        "prefill saved 1.28M",
    ]:
        assert label in html


def test_local_page_has_v2_evidence_cockpit_sections():
    html = _read("index.html")
    for label in [
        "V2.0 Evidence Cockpit",
        "用户态 AgentOS Runtime 原型",
        "Runtime 核心闭环",
        "V2.0 调度智能层",
        "实验验证矩阵",
        "Agent 一等实体",
        "ContextRef 管理",
        "KV/Prefix Cache 亲和",
        "EventLog / Dashboard",
    ]:
        assert label in html


def test_local_page_has_adaptive_scheduler_and_probe_mock_data():
    html = _read("index.html")
    for label in [
        "WorkloadFeature 采样",
        "Workload Classifier",
        "Policy Selector",
        "Fairness / Starvation",
        "scheduler.policy_selected",
        "cache_affinity",
        "sched_ext / eBPF / BPF Map",
        "probe / fallback / mock adapter",
    ]:
        assert label in html
    # V2 decision summary is now dynamically rendered via JS id
    assert 'id="v2-decision-summary"' in html
    assert 'id="v2-decision-detail"' in html


def test_local_page_has_local_only_validation_tracks():
    html = _read("index.html")
    local = _page_section(html, "page-local")
    for label in [
        "本地 LLM",
        "Runtime 基线对比",
        "Pressure Suite",
        "AI 推理",
        "编译压力",
        "压缩压力",
        "网络压测",
    ]:
        assert label in local


def test_kpi_sparklines_have_dedicated_bottom_lane():
    css = _read("style.css")
    assert ".kpi .spark, .cloud-kpi .spark" in css
    assert "bottom: 10px;" in css
    assert "height: 18px;" in css
    assert "padding: 12px 13px 42px;" in css


def test_local_page_uses_two_by_two_core_grid_before_auxiliary_evidence():
    html = _read("index.html")
    css = _read("style.css")
    assert '<div class="local-core-grid">' in html
    assert '<div class="local-secondary-grid">' in html
    assert html.index('id="zone-process"') < html.index('class="local-secondary-grid"')
    assert html.index('id="zone-cache"') < html.index('class="local-secondary-grid"')
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in css
    assert ".local-secondary-grid" in css


def test_local_core_layout_gives_context_and_cache_normal_height():
    css = _read("style.css")
    assert "grid-template-areas:" in css
    assert '"process dag"' in css
    assert '"memory cache"' in css
    assert "repeat(2, 270px)" not in css
    for selector in [
        ".core-panel.proc",
        ".dag-card",
        ".core-panel.mem",
        ".core-panel.cache",
    ]:
        assert selector in css
    assert "min-height: 420px;" in css
    assert "min-height: 500px;" in css


def test_cloud_page_contains_framework_benchmark_matrix():
    html = _read("index.html")
    for label in [
        "Cloud Agent Framework Benchmark",
        "OpenAI Agents SDK",
        "CrewAI",
        "LangGraph",
        "AutoGen",
        "Native Framework 原生运行",
        "+ AgentOS Runtime",
        "Runtime 对比原生框架",
        "Cached Tokens / Context Reuse",
        "真实接入证据",
        "统一观测与控制平面",
    ]:
        assert label in html
    # Google ADK should NOT be in the default framework list
    assert "Google ADK" not in html


def test_appjs_has_core_logic():
    js = _read("app.js")
    assert "processEvent" in js
    assert "renderAll" in js
    assert "EventSource" in js
    assert "loadSnapshot" in js
    assert "switchPage" in js
    assert "renderCloudCharts" in js
    assert "dag-live-svg" in js


def test_appjs_curve_uses_warm_accent():
    js = _read("app.js")
    assert "#14b8a6" in js          # warm-keynote teal in SVG curve
    assert "#39bae6" not in js      # old bloomberg cyan must be gone


def test_appjs_preserves_demo_seed_when_live_events_are_sparse():
    js = _read("app.js")
    assert "var DEMO =" in js
    assert "Object.assign({}, demoBase(DEMO.agents), S.agents)" in js
    assert "Object.assign({}, demoBase(DEMO.tasks), S.tasks)" in js
    assert "demoBase(DEMO.cache.curve)" in js
    assert "switchPage('local');\n  renderAll();\n  loadSnapshot().then(" in js


def test_live_html_backward_compat():
    from agentos.dashboard.live_html import LIVE_DASHBOARD_HTML

    assert LIVE_DASHBOARD_HTML
    assert "AgentOS" in LIVE_DASHBOARD_HTML
    # css and js are inlined back into the single string (no external refs)
    assert "#14b8a6" in LIVE_DASHBOARD_HTML       # css inlined
    assert "Cloud Agent Framework Benchmark" in LIVE_DASHBOARD_HTML
    assert "processEvent" in LIVE_DASHBOARD_HTML   # js inlined
    assert '<link rel="stylesheet" href="/style.css">' not in LIVE_DASHBOARD_HTML
    assert '<script src="/app.js"></script>' not in LIVE_DASHBOARD_HTML


def _page_section(html: str, page_id: str) -> str:
    start = html.index(f'id="{page_id}"')
    next_page = html.find('<section id="page-', start + 1)
    return html[start:] if next_page == -1 else html[start:next_page]


def test_local_page_is_local_runtime_pressure_cockpit_only():
    html = _read("index.html")
    local = _page_section(html, "page-local")

    for label in [
        "Local Runtime Evidence Cockpit",
        "Baseline vs Runtime 结果指标",
        "用户态 AgentOS Runtime 原型",
        "V2.0 Workload-aware 调度闭环",
        "本地实验验证体系",
        "Pressure Suite",
        "AI 推理",
        "编译压力",
        "压缩压力",
        "网络压测",
        "sched_ext / eBPF Probe",
    ]:
        assert label in local

    for cloud_only in [
        "云端 API",
        "OpenAI Agents SDK",
        "CrewAI",
        "LangGraph",
        "AutoGen",
        "Cloud Architecture",
    ]:
        assert cloud_only not in local


def test_local_page_v2_evidence_fields_are_visible():
    html = _read("index.html")
    local = _page_section(html, "page-local")

    for label in [
        "WorkloadFeature",
        "WorkloadClass",
        "PolicySelector",
        "scheduler.decision",
        "scheduler.policy_selected",
        "workload_class",
        "policy",
        "fairness_score",
        "starvation_risk",
        "latency_first",
        "throughput_first",
        "cache_affinity",
        "fairness_balanced",
    ]:
        assert label in local


def test_style_prioritizes_misans_and_has_semantic_cloud_icons():
    css = _read("style.css")

    font_line = next(line for line in css.splitlines() if "--font-ui:" in line)
    assert "MiSans" in font_line
    assert font_line.index("MiSans") < font_line.index("Inter")
    assert "--font-ui:" in css
    assert "font-family: var(--font-ui);" in css

    for selector in [
        ".cloud-icon.grid",
        ".cloud-icon.stack",
        ".cloud-icon.play",
        ".cloud-icon.clock",
        ".cloud-icon.doc",
        ".cloud-icon.cost",
        ".cloud-icon.shield",
        ".cloud-icon.bars",
    ]:
        assert selector in css


def test_local_evidence_layout_is_compact_and_polished():
    html = _read("index.html")
    css = _read("style.css")

    for marker in [
        "runtime-foundation refined-flow",
        "workload-loop refined-flow",
        "loop-feedback",
        "foundation-arrow",
        "compact-density",
    ]:
        assert marker in html or marker in css

    for expected in [
        ".local-evidence-cockpit {",
        "gap: 7px;",
        "min-height: 68px;",
        "min-height: 86px;",
        "grid-template-columns: minmax(0, 1fr) 340px;",
    ]:
        assert expected in css


# --- Phase 2: real data integration tests ---

def test_index_has_kpi_ids_for_real_data():
    """KPI strong/small elements have stable ids for JS to update."""
    html = _read("index.html")
    for dom_id in [
        "kpi-token-saved",
        "kpi-makespan",
        "kpi-cache-hit",
        "kpi-prefill",
        "kpi-recovery",
        "kpi-eventlog",
        "kpi-policy-switches",
        "kpi-fairness",
        "kpi-fairness-small",
    ]:
        assert f'id="{dom_id}"' in html, f"missing KPI id: {dom_id}"


def test_index_has_decision_card_ids():
    """Decision card dd elements have stable ids for JS to update."""
    html = _read("index.html")
    for dom_id in [
        "sched-event",
        "sched-workload-class",
        "sched-policy",
        "sched-reason",
        "sched-fairness",
        "sched-starvation-risk",
    ]:
        assert f'id="{dom_id}"' in html, f"missing decision card id: {dom_id}"


def test_index_has_probe_card_ids():
    """Probe card dd elements have stable ids for JS to update."""
    html = _read("index.html")
    for dom_id in [
        "probe-sched-ext",
        "probe-bpf-map",
        "probe-reason",
        "probe-bpf-small",
    ]:
        assert f'id="{dom_id}"' in html, f"missing probe card id: {dom_id}"


def test_index_has_source_pill_id():
    """Source pill has id for JS to update with real data label."""
    html = _read("index.html")
    assert 'id="local-source-pill"' in html


def test_index_has_pressure_grid_id():
    """Pressure grid has id for JS to render workload matrix."""
    html = _read("index.html")
    assert 'id="pressure-grid"' in html


def test_index_has_data_policy_on_chipset():
    """Policy chipset spans have data-policy attributes."""
    html = _read("index.html")
    for policy in ["latency_first", "throughput_first", "cache_affinity", "fairness_balanced"]:
        assert f'data-policy="{policy}"' in html, f"missing data-policy: {policy}"


def test_appjs_handles_scheduler_events():
    """app.js processes scheduler events for real data."""
    js = _read("app.js")
    assert "scheduler.policy_selected" in js
    assert "scheduler.decision" in js
    assert "scheduler.fairness_warning" in js
    assert "scheduler.kernel_probe" in js
    assert "scheduler.bpf_map_updated" in js


def test_appjs_loads_workload_matrix():
    """app.js fetches /api/workload-matrix for pressure matrix."""
    js = _read("app.js")
    assert "/api/workload-matrix" in js
    assert "loadWorkloadMatrix" in js
    assert "renderPressureMatrix" in js


def test_appjs_has_per_trace_timing():
    """app.js tracks per-trace timing by trace_id prefix."""
    js = _read("app.js")
    assert "tracePrefix" in js
    assert "first_created" in js
    assert "last_completed" in js
    assert "computeLatestMakespan" in js


def test_appjs_updates_source_pill():
    """app.js updates source pill with real data label."""
    js = _read("app.js")
    assert "local-source-pill" in js
    assert "real data" in js


def test_appjs_preserves_demo_fallback():
    """app.js still has DEMO fallback for when no real data is available."""
    js = _read("app.js")
    assert "var DEMO =" in js
    assert "DEMO.agents" in js
    assert "DEMO.tasks" in js
    assert "DEMO.cache.curve" in js


def test_appjs_handles_llm_response_received():
    """app.js still processes llm.response_received for token tracking."""
    js = _read("app.js")
    assert "llm.response_received" in js
    assert "cache_hit_tokens" in js
    assert "cache_miss_tokens" in js


def test_appjs_handles_context_compressed():
    """app.js still processes context.compressed for token saved."""
    js = _read("app.js")
    assert "context.compressed" in js
    assert "tokens_saved" in js


# --- Phase 4: cloud benchmark integration tests ---

def test_index_has_cloud_kpi_ids():
    """Cloud KPI strong elements have stable ids for JS to update."""
    html = _read("index.html")
    for dom_id in [
        "cloud-kpi-frameworks",
        "cloud-kpi-scenarios",
        "cloud-kpi-runs",
        "cloud-kpi-latency",
        "cloud-kpi-tokens",
        "cloud-kpi-cost",
        "cloud-kpi-recovery",
        "cloud-kpi-events",
        "cloud-footer-source",
    ]:
        assert f'id="{dom_id}"' in html, f"missing cloud KPI id: {dom_id}"


def test_index_cloud_uses_crewai_not_google_adk():
    """Cloud page uses CrewAI, not Google ADK."""
    html = _read("index.html")
    assert "CrewAI" in html
    # Google ADK should not appear in architecture or timeline
    local = _page_section(html, "page-cloud")
    assert "Google ADK" not in local


def test_index_cloud_footer_mentions_deepseek():
    """Cloud footer mentions DeepSeek, not gpt-4.1."""
    html = _read("index.html")
    assert "DeepSeek" in html
    assert "gpt-4.1" not in html


def test_appjs_handles_degraded_improvement():
    """app.js handles degraded/empty improvement without NaN/undefined."""
    js = _read("app.js")
    assert "improvementCell" in js
    assert "statusBadge" in js
    assert "n/a" in js  # fallback for empty improvement
    assert "fmtCost" in js  # safe cost formatting


def test_appjs_cloud_kpi_update():
    """app.js updates cloud KPIs from payload metadata."""
    js = _read("app.js")
    assert "cloud-kpi-frameworks" in js
    assert "cloud-kpi-scenarios" in js
    assert "cloud-kpi-runs" in js
    assert "cloud-kpi-latency" in js
    assert "cloud-kpi-tokens" in js
    assert "cloud-kpi-cost" in js
    assert "cloud-kpi-recovery" in js
    assert "cloud-kpi-events" in js
    assert "cloud-footer-source" in js


def test_appjs_cloud_chart_handles_missing_data():
    """app.js cloud charts filter rows with missing/zero data."""
    js = _read("app.js")
    assert "renderInsightCards" in js


def test_index_has_insight_cards():
    """HTML has insight card sections for the three conclusion cards."""
    html = _read("index.html")
    for dom_id in [
        "insight-latency",
        "insight-prompt",
        "insight-cost",
        "insight-latency-list",
        "insight-prompt-list",
        "insight-cost-list",
        "recovery-card",
        "recovery-matrix",
    ]:
        assert f'id="{dom_id}"' in html, f"missing insight card id: {dom_id}"


def test_index_insight_cards_have_chinese_titles():
    """Insight cards have Chinese titles, not old English ones."""
    html = _read("index.html")
    assert "平均延迟变化" in html
    assert "Prompt 开销与上下文复用" in html
    assert "API 成本变化" in html
    assert "统一故障恢复验证" in html
    assert "人民币分" not in html
    assert "原始美元值保留在 Matrix" not in html
    # Old English titles absent
    assert "Latency by Framework" not in html
    assert "Prompt Tokens / Cached Tokens" not in html
    assert "Cost per Scenario" not in html
    assert "Fault Injection & Recovery Timeline" not in html


def test_index_cloud_architecture_chinese():
    """Cloud Architecture section is Chinese with vertical flow."""
    html = _read("index.html")
    assert "云端接入架构" in html
    assert "保留框架编排，统一接入 Runtime" in html
    assert "框架适配层" in html
    assert "AgentOS Runtime 层" in html
    assert "云端模型后端" in html
    assert "统一观测与控制平面" in html
    # Vertical flow
    assert "arch-flow-vertical" in html
    assert "arch-stage" in html
    assert "arch-arrow" in html
    assert "control-plane-strip" in html
    # Old English absent
    assert "Cloud Architecture" not in html
    assert "AgentOS Runtime in the Cloud" not in html
    assert "Unified Observability & Control Plane" not in html


def test_css_has_vertical_arch_flow():
    """CSS has vertical layout rules for arch-flow."""
    css = _read("style.css")
    assert "arch-flow-vertical" in css
    assert "flex-direction: column" in css
    assert "justify-content: space-around" in css
    assert "arch-stage" in css
    assert "arch-arrow" in css
    assert "control-plane-strip" in css


def test_index_has_evidence_grid_layout():
    """Bottom area uses cloud-evidence-grid layout."""
    html = _read("index.html")
    assert "cloud-evidence-grid" in html
    assert "cloud-architecture-panel" in html
    assert "real-evidence-panel" in html
    assert "latency-card" in html
    assert "prompt-card" in html
    assert "cost-card" in html
    assert "recovery-card" in html
    # Evidence list is now inside the grid, not a sidebar
    assert "evidence-list" in html
    assert "evidence-item" in html


def test_index_has_prompt_legend():
    """Prompt card has legend explaining orange bar and reuse badge."""
    html = _read("index.html")
    assert "legend-dot" in html or "legend-pill" in html
    assert "Prompt Token 开销" in html
    assert "上下文复用" in html


def test_appjs_renders_insight_cards():
    """app.js renders three insight cards from benchmark data."""
    js = _read("app.js")
    assert "renderInsightCards" in js
    assert "insight-latency-list" in js
    assert "insight-prompt-list" in js
    assert "insight-cost-list" in js
    assert "fmtRmbFen" not in js
    assert "USD_TO_CNY_DISPLAY_RATE" not in js
    assert "基本持平" not in js
    assert "0.0%" in js
    assert "delta-good" in js
    assert "delta-bad" in js
    # Comparison bars
    assert "comparison-row" in js
    assert "bar-line" in js
    assert "bar-pair" in js
    # Prompt card
    assert "reuse-badge" in js
    assert "框架模板较长" in js
    # Recovery matrix
    assert "recovery-matrix" in js
    assert "recovery-row" in js
    assert "recovery-badge" in js
    assert "已注入" in js
    assert "已恢复" in js
    # Data sources
    assert "avg_latency" in js
    assert "api_cost" in js
    assert "prompt_tokens" in js
    assert "context_reuse" in js
    assert "recovered_faults" in js
    assert "improvement" in js


def test_index_has_evidence_rail_ids():
    """Evidence rail small elements have ids for dynamic update."""
    html = _read("index.html")
    for dom_id in [
        "evidence-scheduling-small",
        "evidence-reuse-small",
        "evidence-cost-small",
        "evidence-fault-small",
        "evidence-provenance-small",
    ]:
        assert f'id="{dom_id}"' in html, f"missing evidence rail id: {dom_id}"


def test_appjs_dynamically_updates_evidence_rail():
    """app.js dynamically updates evidence rail from JSON payload."""
    js = _read("app.js")
    assert "evidence-scheduling-small" in js
    assert "evidence-cost-small" in js
    assert "evidence-fault-small" in js
    assert "evidence-provenance-small" in js
    assert "framework_orchestrated_runtime" in js
    assert "strictRow" in js or "strict_framework" in js


def test_index_no_hardcoded_latency_percentages():
    """HTML does not contain hardcoded benchmark latency percentages."""
    html = _read("index.html")
    for bad in ["-43.8%", "-42.4%", "-40.4%", "-40.7%", "-79.5%", "-80.8%"]:
        assert bad not in html, f"HTML contains hardcoded latency: {bad}"


def test_index_no_gateway_direct_hardcoding():
    """HTML does not hardcode 1/4 gateway direct text."""
    html = _read("index.html")
    assert "1/4" not in html or "gateway direct" not in html.lower()


def test_appjs_dynamic_strict_runtime():
    """app.js dynamically shows 4/4 strict runtime when all frameworks qualify."""
    js = _read("app.js")
    assert "framework_orchestrated_runtime_count" in js
    assert "四个框架均保留原生编排" in js


def test_framework_icons_have_text_labels():
    """Framework icons use letter labels, not just empty spans."""
    js = _read("app.js")
    assert "fwLetter" in js
    # Check that fwLetter returns distinctive letters
    assert "'O'" in js  # OpenAI
    assert "'L'" in js  # LangGraph
    assert "'A'" in js  # AutoGen
    assert "'C'" in js  # CrewAI
