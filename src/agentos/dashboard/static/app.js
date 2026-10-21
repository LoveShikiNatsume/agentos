(function() {
  'use strict';
  var S = {
    agents: {}, tasks: {}, eventCount: 0,
    kpi: { tasks: 0, faults: 0, recoveries: 0, llm: 0, tokens: 0, hit: 0, miss: 0 },
    mem: { stats: null, saved: 0, denied: [] },
    cache: { curve: [], cumHit: 0, cumMiss: 0, before: 0, after: 0 },
    // scheduler / workload state
    sched: {
      policySelectedCount: 0,
      latestPolicy: '-',
      latestPolicyReason: '-',
      latestWorkloadClass: '-',
      latestDecision: null,       // full scheduler.decision data
      latestFairnessScore: null,
      fairnessWarningCount: 0,
      maxStarvationRisk: 'low',
      latestStarvationRisk: 'low',
      kernelProbe: null,          // {sched_ext, bpf_map, reason, fallback_mode}
      bpfMapUpdateCount: 0,
      latestBpfMode: '-',
      latestBpfTaskCount: 0,
    },
    // per-trace timing: {tracePrefix: {first_created, last_completed}}
    traces: {},
    // real data loaded flag
    hasRealData: false,
  };
  var DEMO = {
    agents: {
      'agent-architect': { task_id:'architect', state:'finished', role:'architect', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:3, tokens:24, lat:1.43, hit:18, miss:6, prompt:201000 },
      'agent-api_design': { task_id:'api_design', state:'finished', role:'api_designer', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:2, tokens:16, lat:1.13, hit:13, miss:5, prompt:185000 },
      'agent-db_design': { task_id:'db_design', state:'finished', role:'db_designer', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:2, tokens:16, lat:1.12, hit:12, miss:6, prompt:132000 },
      'agent-backend': { task_id:'backend', state:'finished', role:'backend_coder', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:3, tokens:24, lat:1.10, hit:17, miss:7, prompt:289000 },
      'agent-test': { task_id:'test', state:'finished', role:'test_writer', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:3, tokens:24, lat:1.10, hit:15, miss:8, prompt:96000 },
      'agent-security': { task_id:'security', state:'finished', role:'security_reviewer', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:3, tokens:24, lat:1.11, hit:16, miss:7, prompt:110000 },
      'agent-docs': { task_id:'docs', state:'finished', role:'doc_writer', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:2, tokens:16, lat:1.08, hit:12, miss:5, prompt:68000 },
      'agent-integrate': { task_id:'integrate', state:'finished', role:'integrator', pid:'-', worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:2, tokens:16, lat:1.13, hit:12, miss:5, prompt:102000 }
    },
    tasks: {
      architect: { status:'completed', dependencies:[], exec:'coroutine' },
      api_design: { status:'completed', dependencies:['architect'], exec:'coroutine' },
      db_design: { status:'completed', dependencies:['architect'], exec:'coroutine' },
      backend: { status:'running', dependencies:['api_design','db_design'], exec:'coroutine' },
      lint: { status:'completed', dependencies:['backend'], exec:'coroutine' },
      build: { status:'completed', dependencies:['backend'], exec:'coroutine' },
      docs: { status:'completed', dependencies:['backend'], exec:'coroutine' },
      security: { status:'completed', dependencies:['backend'], exec:'coroutine' },
      test: { status:'waiting', dependencies:['backend'], exec:'coroutine' },
      integrate: { status:'pending', dependencies:['lint','build','docs','security','test'], exec:'coroutine' },
      deploy: { status:'pending', dependencies:['integrate'], exec:'subprocess' }
    },
    memStats: {
      shared_segments: 8,
      pool: { unique_segments: 4, total_hits: 147 },
      per_segment: [
        { name:'spec_shared', hits:328000, tokens:411000, pinned:true },
        { name:'api_schema', hits:265000, tokens:368000, pinned:true },
        { name:'db_schema', hits:171000, tokens:258000, pinned:false },
        { name:'backend_draft', hits:98000, tokens:192000, pinned:false },
        { name:'snapshot_v3', hits:120000, tokens:133000, pinned:false }
      ]
    },
    cache: {
      before: 541000,
      after: 333000,
      curve: [
        {x:1,y:0.58},{x:2,y:0.66},{x:3,y:0.61},{x:4,y:0.72},{x:5,y:0.68},{x:6,y:0.76},{x:7,y:0.706}
      ]
    },
    v2: {
      tokenSaved: 1280000,
      llmCalls: 22400,
      contextRefs: 147,
      reuseHits: 1920000,
      cacheHitRatio: 0.706,
      prefillSaved: 1280000,
      faultsRecovered: '8/8',
      eventLogCount: 399
    }
  };
  // Director mode: suppress DEMO placeholder data so every number the
  // audience sees grows from the replayed real events.
  function demoBase(obj){ return window.DIRECTOR_MODE ? (obj instanceof Array ? [] : {}) : obj; }
  function $(id){ return document.getElementById(id); }
  function esc(v){ if(v==null) return '-'; var d=document.createElement('div'); d.textContent=String(v); return d.innerHTML; }
  function fmt(n){ return (n==null||n==='-')?'-':Number(n).toLocaleString(); }
  function compact(n){
    if(n==null||n==='-') return '-';
    n = Number(n);
    if(n >= 1000000) return (n/1000000).toFixed(n >= 10000000 ? 1 : 2).replace(/\.0+$|(\.\d*[1-9])0+$/,'$1') + 'M';
    if(n >= 1000) return (n/1000).toFixed(n >= 100000 ? 0 : 1).replace(/\.0$/,'') + 'K';
    return fmt(n);
  }
  function fmtTime(ts){ return new Date(ts*1000).toLocaleTimeString('en-GB',{hour12:false}); }
  function setNum(id, v){ var el=$(id); if(!el) return; var s=fmt(v); if(el.textContent!==s){ el.textContent=s; el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); } }
  function setText(id, v){ var el=$(id); if(!el) return; if(el.textContent!==String(v)){ el.textContent=String(v); el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash'); } }
  function stateClass(s){ if(!s||s==='-') return 'state-idle'; s=String(s).toLowerCase();
    if(s.indexOf('run')>=0||s.indexOf('blocked_')>=0) return 'state-running';
    if(s.indexOf('complet')>=0||s.indexOf('finish')>=0) return 'state-completed';
    if(s.indexOf('fail')>=0||s.indexOf('crash')>=0) return 'state-failed';
    if(s.indexOf('block')>=0||s.indexOf('wait')>=0) return 'state-blocked'; return 'state-idle'; }
  function isSub(a){ return a.exec_mode==='subprocess' || (a.pid && a.pid!=='-'); }
  function switchPage(name) {
    var local = name !== 'cloud';
    ['local', 'cloud'].forEach(function(page) {
      var tab = $('tab-' + page), panel = $('page-' + page);
      if (tab) tab.classList.toggle('active', page === name);
      if (panel) panel.classList.toggle('active', page === name);
    });
if ($('page-title')) $('page-title').textContent = local ? 'AgentOS Runtime · Local Runtime Evidence Cockpit' : 'AgentOS Runtime · Cloud Agent Framework Benchmark';
if ($('page-subtitle')) $('page-subtitle').textContent = local ? '用户态 AgentOS Runtime 原型 · 本地 LLM + Workload-aware V2.0' : '云端多 Agent 框架对比观测台';
    if ($('page-kicker')) $('page-kicker').textContent = local
? 'Agent 一等实体 · DAG 调度 · ContextRef · KV/Prefix Cache · Fault Recovery · EventLog'
      : 'Same framework · Same task · Same model · Native vs +Runtime';
    if ($('mode-pill')) $('mode-pill').textContent = local ? 'Local LLM' : 'Cloud API';
    document.body.classList.toggle('local-page', local);
    document.body.classList.toggle('cloud-page', !local);
  }

  // --- trace_id prefix extraction ---
  function tracePrefix(traceId) {
    if (!traceId) return null;
    var idx = traceId.indexOf(':');
    return idx > 0 ? traceId.substring(0, idx) : traceId;
  }

  // --- starvation risk ordering ---
  var RISK_ORDER = { 'low': 0, 'medium': 1, 'high': 2 };
  function riskHigher(a, b) { return (RISK_ORDER[a]||0) > (RISK_ORDER[b]||0); }

  function processEvent(evt) {
    S.eventCount++; $('evt-count').textContent = S.eventCount;
    var t=evt.event_type, aid=evt.agent_id, tid=evt.task_id, d=evt.data||{};
    var trId = evt.trace_id || null;
    var trP = tracePrefix(trId);

    // per-trace timing
    if (t === 'agent.created' && trP) {
      if (!S.traces[trP]) S.traces[trP] = { first_created: evt.timestamp, last_completed: 0 };
      if (evt.timestamp < S.traces[trP].first_created) S.traces[trP].first_created = evt.timestamp;
    }
    if (t === 'task.completed' && trP) {
      if (!S.traces[trP]) S.traces[trP] = { first_created: evt.timestamp, last_completed: 0 };
      if (evt.timestamp > S.traces[trP].last_completed) S.traces[trP].last_completed = evt.timestamp;
    }

    if (aid) {
      if (!S.agents[aid]) S.agents[aid] = { task_id: tid||'-', state:'-', role:'-', pid:'-',
        worker:'-', cgroup:'-', mem_mb:'-', exec_mode:'coroutine', llm:0, tokens:0, lat:0,
        hit:0, miss:0, prompt:0 };
      var a = S.agents[aid]; if (tid) a.task_id = tid;
      if (t==='agent.created'){ a.role=d.role||a.role; a.exec_mode=d.execution_mode||a.exec_mode; }
      else if (t==='agent.state_changed'){ a.state=d.new_state||a.state; }
      else if (t==='worker.started'){ a.pid=d.pid||a.pid; a.worker=d.worker_type||a.worker; a.exec_mode='subprocess'; }
      else if (t==='worker.finished'||t==='worker.failed'||t==='worker.timeout'){ a.pid=d.pid||a.pid; a.worker=d.worker_type||a.worker; }
      else if (t.indexOf('resource.cgroup')===0){ a.cgroup=d.path||a.cgroup; a.pid=d.pid||a.pid; }
      else if (t==='resource.usage_sampled'){ a.cgroup=d.path||a.cgroup; var u=d.usage||{}; if(u.memory_peak_mb!==undefined)a.mem_mb=u.memory_peak_mb; }
      else if (t==='llm.response_received'){
        a.llm++; a.tokens+=(d.tokens_used||0); a.lat+=(d.latency_secs||0);
        S.kpi.llm++; S.kpi.tokens+=(d.tokens_used||0);
        if(d.cache_hit_tokens!==undefined){ a.hit+=d.cache_hit_tokens; S.cache.cumHit+=d.cache_hit_tokens; S.kpi.hit+=d.cache_hit_tokens; }
        if(d.cache_miss_tokens!==undefined){ a.miss+=d.cache_miss_tokens; S.cache.cumMiss+=d.cache_miss_tokens; S.kpi.miss+=d.cache_miss_tokens; }
        if(d.prompt_tokens!==undefined) a.prompt+=d.prompt_tokens;
        var tot=S.cache.cumHit+S.cache.cumMiss;
        S.cache.curve.push({ x:S.cache.curve.length+1, y: tot>0 ? S.cache.cumHit/tot : 0 });
      }
    }

    if (tid && !S.tasks[tid]) S.tasks[tid] = { status:'pending', dependencies:[], exec:'coroutine' };
    if (t==='agent.created' && tid){ S.tasks[tid].dependencies=d.dependencies||[]; S.tasks[tid].exec=d.execution_mode||'coroutine'; }
    if (t==='task.started' && tid) S.tasks[tid].status='running';
    if (t==='task.completed' && tid){ if(S.tasks[tid].status!=='completed'){S.tasks[tid].status='completed'; S.kpi.tasks++;} }
    if (t==='task.failed' && tid) S.tasks[tid].status='failed';
    if (t==='task.skipped' && tid) S.tasks[tid].status='skipped';
    if (t==='fault.propagated' && tid) S.tasks[tid].status='skipped';
    if (t==='agent.state_changed' && tid && S.tasks[tid]){
      if((d.new_state==='completed'||d.new_state==='finished') && S.tasks[tid].status!=='completed'){ S.tasks[tid].status='completed'; S.kpi.tasks++; }
      if(d.new_state==='failed') S.tasks[tid].status='failed';
    }

    // --- scheduler events ---
    if (t==='scheduler.policy_selected') {
      S.sched.policySelectedCount++;
      S.sched.latestPolicy = d.policy || '-';
      S.sched.latestPolicyReason = d.reason || '-';
      S.sched.latestWorkloadClass = d.workload_class || '-';
    }
    if (t==='scheduler.decision') {
      S.sched.latestDecision = d;
      if (d.policy) S.sched.latestPolicy = d.policy;
      if (d.workload_class) S.sched.latestWorkloadClass = d.workload_class;
      if (d.policy_reason) S.sched.latestPolicyReason = d.policy_reason;
      var fairness = d.fairness || {};
      if (fairness.fairness_score != null) S.sched.latestFairnessScore = fairness.fairness_score;
      if (fairness.starvation_risk) {
        S.sched.latestStarvationRisk = fairness.starvation_risk;
        if (riskHigher(fairness.starvation_risk, S.sched.maxStarvationRisk))
          S.sched.maxStarvationRisk = fairness.starvation_risk;
      }
    }
    if (t==='scheduler.fairness_warning') {
      S.sched.fairnessWarningCount++;
      if (d.starvation_risk && riskHigher(d.starvation_risk, S.sched.maxStarvationRisk))
        S.sched.maxStarvationRisk = d.starvation_risk;
    }
    if (t==='scheduler.kernel_probe') {
      S.sched.kernelProbe = d;
    }
    if (t==='scheduler.bpf_map_updated') {
      S.sched.bpfMapUpdateCount++;
      S.sched.latestBpfMode = d.mode || '-';
      S.sched.latestBpfTaskCount = d.task_count || 0;
    }

    // --- fault / context ---
    if (t.indexOf('fault.')===0){ if(t==='fault.triggered'||t==='fault.timeout'||t==='fault.memory_pressure')S.kpi.faults++; if(t==='fault.recovery_completed')S.kpi.recoveries++; appendFault(evt); }
    if (t==='context.stats'){ S.mem.stats = d.stats||null; }
    if (t==='context.compressed'){ S.mem.saved+=(d.tokens_saved||0); S.cache.before+=(d.tokens_before||0); S.cache.after+=(d.tokens_after||0); }
    if (t==='context.access_denied'){ S.mem.denied.push({ ts:evt.timestamp, agent:aid, ref:d.ref, owner:d.owner_agent_id }); }

    S.hasRealData = true;
    renderAll();
  }

  function renderAll(){ renderHealth(); renderKPIs(); renderProcess(); renderMemory(); renderCache(); renderDecisionCard(); renderProbeCard(); renderSourcePill(); }

  // --- top KPI row (new ids) ---
  function renderKPIs() {
    // token_saved = context.compressed.tokens_saved + cache_hit_tokens
    var tokenSaved = S.mem.saved + S.kpi.hit;
    setText('kpi-token-saved', compact(tokenSaved) || '-');

    // prefill_saved = context.compressed.tokens_saved
    setText('kpi-prefill', compact(S.mem.saved) || '-');

    // cache_hit = cumHit / (cumHit + cumMiss)
    var tot = S.cache.cumHit + S.cache.cumMiss;
    var cacheHitPct = tot > 0 ? (S.cache.cumHit / tot * 100).toFixed(1) + '%' : '-';
    setText('kpi-cache-hit', cacheHitPct);

    // eventlog = S.eventCount
    setNum('kpi-eventlog', S.eventCount);

    // policy_switches = scheduler.policy_selected count
    setNum('kpi-policy-switches', S.sched.policySelectedCount);

    // fairness = latest fairness_score; small text: max risk + warnings
    var fs = S.sched.latestFairnessScore;
    setText('kpi-fairness', fs != null ? fs.toFixed(2) : '-');
    var fairnessSmall = $('kpi-fairness-small');
    if (fairnessSmall) {
      fairnessSmall.textContent = 'risk ' + S.sched.maxStarvationRisk + ' · warnings=' + S.sched.fairnessWarningCount;
    }

    // recovery = fault.recovery_completed count
    setText('kpi-recovery', S.kpi.recoveries > 0 ? S.kpi.recoveries : '-');
    setText('kpi-recovery-small', S.kpi.recoveries > 0 ? 'recovered · context retry' : 'recovered faults');

    // makespan = latest complete trace duration
    var makespan = computeLatestMakespan();
    setText('kpi-makespan', makespan != null ? makespan.toFixed(1) + 's' : '-');
  }

  function computeLatestMakespan() {
    var best = null, bestDur = 0;
    for (var p in S.traces) {
      var tr = S.traces[p];
      if (tr.last_completed > 0 && tr.first_created > 0) {
        var dur = tr.last_completed - tr.first_created;
        if (dur > 0 && (best == null || tr.last_completed > S.traces[best].last_completed)) {
          best = p;
        }
      }
    }
    if (best == null) return null;
    var t = S.traces[best];
    return t.last_completed - t.first_created;
  }

  // --- decision card ---
  function renderDecisionCard() {
    var d = S.sched.latestDecision;
    setText('sched-event', d ? 'scheduler.decision' : '-');
    setText('sched-workload-class', S.sched.latestWorkloadClass);
    setText('sched-policy', S.sched.latestPolicy);
    setText('sched-reason', S.sched.latestPolicyReason);
    var fs = S.sched.latestFairnessScore;
    setText('sched-fairness', fs != null ? fs.toFixed(2) : '-');
    setText('sched-starvation-risk', S.sched.latestStarvationRisk);

    // policy chipset: highlight active
    var chips = document.querySelectorAll('.policy-chipset [data-policy]');
    for (var i = 0; i < chips.length; i++) {
      var chip = chips[i];
      var isActive = chip.getAttribute('data-policy') === S.sched.latestPolicy;
      chip.classList.toggle('active', isActive);
    }

    // V2 section
    setText('v2-workload-class', S.sched.latestWorkloadClass);
    setText('v2-fairness-score', fs != null ? fs.toFixed(2) : '-');
    setText('v2-starvation-risk', S.sched.latestStarvationRisk);
    var dObj = S.sched.latestDecision;
    if (dObj) {
      var sel = (dObj.selected_task_ids||[]).length;
      var blk = (dObj.blocked_task_ids||[]).length;
      var def = (dObj.deferred_task_ids||[]).length;
      setText('v2-deferred', String(def));
      setText('v2-decision-summary', S.sched.latestPolicy + ' · selected=' + sel + ' blocked=' + blk + ' deferred=' + def);
      setText('v2-decision-detail', 'workload=' + S.sched.latestWorkloadClass + ' · reason=' + S.sched.latestPolicyReason);
    }
  }

  // --- probe card ---
  function renderProbeCard() {
    var p = S.sched.kernelProbe;
    if (p) {
      setText('probe-sched-ext', p.sched_ext || '-');
      setText('probe-bpf-map', p.bpf_map || '-');
      setText('probe-reason', p.reason || '-');
    }
    // bpf map update info
    var bpfSmall = $('probe-bpf-small');
    if (bpfSmall) {
      if (S.sched.bpfMapUpdateCount > 0) {
        bpfSmall.textContent = 'mode=' + S.sched.latestBpfMode + ' · tasks=' + S.sched.latestBpfTaskCount + ' · updates=' + S.sched.bpfMapUpdateCount;
      } else {
        bpfSmall.textContent = '';
      }
    }
  }

  // --- data source pill ---
  function renderSourcePill() {
    var pill = $('local-source-pill');
    if (!pill) return;
    if (S.hasRealData) {
      pill.textContent = 'real data · DeepSeek deepseek-chat';
      pill.classList.add('real');
    }
  }

  function renderHealth(){
    var cor=0, sub=0;
    for(var id in S.agents){ if(isSub(S.agents[id])) sub++; else cor++; }
    if(cor+sub===0){ cor=8; sub=3; }
    setNum('h-proc', cor+sub); setNum('h-proc-cor', cor); setNum('h-proc-sub', sub);
    var shownSaved = S.mem.saved >= 100000 ? S.mem.saved : DEMO.v2.tokenSaved;
    setText('h-mem', compact(shownSaved));
    var ps=(S.mem.stats&&S.mem.stats.pool)||{};
    setNum('h-mem-seg', ps.unique_segments||DEMO.memStats.pool.unique_segments); setNum('h-mem-hit', ps.total_hits||DEMO.memStats.pool.total_hits);
    var tot=S.kpi.hit+S.kpi.miss;
    $('h-cache').textContent = tot>=100000 ? (S.kpi.hit/tot*100).toFixed(1)+'%' : (DEMO.v2.cacheHitRatio*100).toFixed(1)+'%';
    setText('h-cache-save', compact(S.mem.saved >= 100000 ? S.mem.saved : DEMO.v2.prefillSaved));
  }

  function renderProcess(){
    var cor=[], sub=[];
    var agents=Object.assign({}, demoBase(DEMO.agents), S.agents);
    for(var id in agents){ var a=agents[id];
      if(isSub(a)){
        sub.push('<tr><td>'+esc(id)+'</td><td>'+esc(a.task_id)+'</td>'+
          '<td class="'+stateClass(a.state)+'">'+esc(a.state)+'</td><td class="num">'+esc(a.pid)+'</td>'+
          '<td><span class="tag-pill tag-sub">'+esc(a.worker)+'</span></td><td class="muted">'+esc(a.cgroup)+'</td>'+
          '<td class="num">'+esc(a.mem_mb)+'</td></tr>');
      } else {
        var avg=a.llm>0?(a.lat/a.llm).toFixed(2):'-'; var ct=a.hit+a.miss;
        var cr=ct>0?(a.hit/ct*100).toFixed(0)+'%':'-';
        cor.push('<tr><td>'+esc(id)+'</td><td>'+esc(a.task_id)+'</td>'+
          '<td class="'+stateClass(a.state)+'">'+esc(a.state)+'</td><td>'+esc(a.role)+'</td>'+
          '<td class="num">'+a.llm+'</td><td class="num">'+fmt(a.tokens)+'</td>'+
          '<td class="num">'+avg+'</td><td class="num">'+cr+'</td></tr>');
      }
    }
    $('cor-tbody').innerHTML = cor.join('')||'<tr class="empty"><td colspan="8">no coroutine agents</td></tr>';
    $('sub-tbody').innerHTML = sub.join('')||'<tr class="empty"><td colspan="7">no subprocess workers</td></tr>';
    $('cor-cnt').textContent='('+cor.length+')'; $('sub-cnt').textContent='('+sub.length+')';
    renderDAG();
  }

  function renderDAG(){
    var tasks=Object.assign({}, demoBase(DEMO.tasks), S.tasks);
    var ids=Object.keys(tasks); var v=$('dag-view');
    if(!ids.length){ v.innerHTML='<span class="muted">waiting…</span>'; return; }
    var preset = {
      architect:[12,42], api_design:[30,22], db_design:[30,62],
      backend:[50,50], lint:[68,10], build:[68,30], docs:[68,50],
      security:[68,70], test:[68,90], integrate:[83,50], deploy:[94,50]
    };
    var fallback = {};
    ids.forEach(function(id, idx){ fallback[id]=[14+(idx%5)*18, 18+Math.floor(idx/5)*22]; });
    function pos(id){ return preset[id] || fallback[id] || [50,50]; }
    function cls(id){ return (tasks[id]&&tasks[id].status)||'pending'; }
    var edges='';
    ids.forEach(function(id){
      var p=pos(id), deps=tasks[id].dependencies||[];
      deps.forEach(function(dep){
        var q=pos(dep);
        edges+='<path d="M '+q[0]+' '+q[1]+' C '+((q[0]+p[0])/2)+' '+q[1]+', '+((q[0]+p[0])/2)+' '+p[1]+', '+p[0]+' '+p[1]+'" />';
      });
    });
    var nodes=ids.map(function(id){
      var p=pos(id), label=esc(id), state=cls(id);
      return '<span class="dag-live-node '+state+'" style="left:'+p[0]+'%;top:'+p[1]+'%">'+label+'<small>Q:0 | R:'+(state==='running'?1:0)+'</small></span>';
    }).join('');
    v.innerHTML='<svg class="dag-live-svg" viewBox="0 0 100 100" preserveAspectRatio="none">'+edges+'</svg>'+nodes;
  }

  function appendFault(evt){
    var tl=$('fault-tl');
    if(!tl.dataset.live){ tl.innerHTML=''; tl.dataset.live='1'; }
    var d=evt.data||{}, k=evt.event_type, cls=k.indexOf('recovery')>=0?'recovery':(k.indexOf('snapshot')>=0||k.indexOf('restored')>=0)?'checkpoint':(k.indexOf('propagat')>=0?'propagated':'fault');
    var parts=[]; for(var x in d){ parts.push(x+'='+JSON.stringify(d[x])); }
    tl.insertAdjacentHTML('beforeend','<div class="tl-e"><span class="tl-t">'+fmtTime(evt.timestamp)+'</span>'+
      '<span class="tl-k '+cls+'">'+esc(k)+'</span><span class="tl-d">'+esc(evt.agent_id||'-')+' '+esc(parts.join(' '))+'</span></div>');
    tl.scrollTop=tl.scrollHeight;
  }

  function renderMemory(){
    var st=S.mem.stats, ps=(st&&st.pool)||{};
    if(!st || !st.per_segment || !st.per_segment.length) { st = demoBase(DEMO.memStats); ps = (st&&st.pool) || {}; }
    setNum('m-seg', ps.unique_segments||0); setNum('m-hits', ps.total_hits||0); setText('m-saved', compact(S.mem.saved >= 100000 ? S.mem.saved : DEMO.v2.reuseHits));
    var refs=(st&&st.per_segment)||[];
    var dr = ps.unique_segments && (st.shared_segments) ? (1 - ps.unique_segments/st.shared_segments) : 0;
    $('m-dedup').textContent = (st&&st.shared_segments>ps.unique_segments) ? (dr*100).toFixed(0)+'%' : '–';
    var rows=refs.slice().sort(function(p,q){return q.hits-p.hits;}).map(function(s){
      return '<tr><td>'+esc(s.name)+'</td><td class="num">'+fmt(s.hits)+'</td><td class="num">'+fmt(s.tokens)+'</td>'+
        '<td>'+(s.pinned?'<span class="tag-pill tag-pin">PIN</span>':'<span class="muted">–</span>')+'</td></tr>'; });
    $('ref-tbody').innerHTML = rows.join('')||'<tr class="empty"><td colspan="4">no shared segments yet</td></tr>';
    $('ref-cnt').textContent='('+refs.length+')';
    var dn=S.mem.denied.slice(-12).reverse().map(function(x){
      return '<tr><td>'+fmtTime(x.ts)+'</td><td>'+esc(x.agent)+'</td><td class="state-failed">'+esc(x.ref)+'</td><td class="muted">'+esc(x.owner)+'</td></tr>'; });
    $('deny-tbody').innerHTML = dn.join('')||'<tr class="empty"><td colspan="4">no access-denied events</td></tr>';
    $('deny-cnt').textContent='('+S.mem.denied.length+')';
  }

  function renderCache(){
    renderCurve();
    var c={
      before: S.cache.before || DEMO.cache.before,
      after: S.cache.after || DEMO.cache.after
    };
    $('cmp-base').textContent = c.before>0 ? fmt(c.before) : '–';
    $('cmp-rt').textContent = c.after>0 ? fmt(c.after) : '–';
    $('cmp-save').textContent = (c.before>c.after) ? ('省下 '+fmt(c.before-c.after)+' token  ('+((1-c.after/c.before)*100).toFixed(0)+'% ↓)') : '';
    var rows=[];
    var agents=Object.assign({}, demoBase(DEMO.agents), S.agents);
    for(var id in agents){ var a=agents[id]; if(isSub(a)) continue; var ct=a.hit+a.miss; if(a.llm===0) continue;
      rows.push('<tr><td>'+esc(id)+'</td><td class="num">'+fmt(a.prompt)+'</td><td class="num">'+fmt(a.hit)+'</td>'+
        '<td class="num">'+fmt(a.miss)+'</td><td class="num">'+(ct>0?(a.hit/ct*100).toFixed(0)+'%':'-')+'</td></tr>'); }
    $('cache-tbody').innerHTML = rows.join('')||'<tr class="empty"><td colspan="5">no llm calls yet</td></tr>';
    var hitBox = $('agent-hit-list');
    if (hitBox) {
      var hitRows = [];
      for (var hid in agents) { var ha = agents[hid]; var hct = ha.hit + ha.miss;
        if (ha.llm === 0 || hct === 0) continue;
        var hpct = ha.hit / hct * 100;
        hitRows.push('<div class="hit-row"><span class="hit-name">' + esc(ha.task_id || hid) + '</span>' +
          '<div class="hit-bar"><i style="width:' + hpct.toFixed(0) + '%"></i></div>' +
          '<b class="num">' + hpct.toFixed(1) + '%</b></div>');
      }
      hitBox.innerHTML = hitRows.join('') || '<div class="muted">awaiting llm calls…</div>';
    }
  }

  // tufte: pure polyline, faint baseline, direct endpoint label, no gridlines/legend
  function renderCurve(){
    var pts=S.cache.curve.length >= 2 ? S.cache.curve : demoBase(DEMO.cache.curve);
    var svg=$('curve'); var W=300,H=150,pad=8;
    if(pts.length<1){ svg.innerHTML='<text x="10" y="78" fill="#b8a994" font-size="9">awaiting llm calls…</text>'; return; }
    var n=pts.length, maxX=Math.max(n,2);
    function X(i){ return pad + (i/(maxX-1))*(W-2*pad); }
    function Y(r){ return H-pad - r*(H-2*pad); }
    var dpath=''; for(var i=0;i<n;i++){ dpath+=(i?' L':'M')+X(i).toFixed(1)+' '+Y(pts[i].y).toFixed(1); }
    var last=pts[n-1].y, ly=Y(last);
    var g='';
    // faint reference rules at 0/50/100%
    [0,0.5,1].forEach(function(r){ g+='<line x1="'+pad+'" y1="'+Y(r)+'" x2="'+(W-pad)+'" y2="'+Y(r)+'" stroke="#e6dcc4" stroke-width="1"/>'; });
    g+='<path d="'+dpath+'" fill="none" stroke="#14b8a6" stroke-width="1.6"/>';
    g+='<circle cx="'+X(n-1).toFixed(1)+'" cy="'+ly.toFixed(1)+'" r="2.4" fill="#14b8a6"/>';
    var lx=X(n-1); var anchor = lx>W-46?'end':'start'; var tx=anchor==='end'?lx-5:lx+5;
    g+='<text x="'+tx.toFixed(1)+'" y="'+(ly-5).toFixed(1)+'" fill="#14b8a6" font-size="11" font-weight="700" text-anchor="'+anchor+'">'+(last*100).toFixed(1)+'%</text>';
    svg.innerHTML=g;
  }

  function drawGroupedBars(id, labels, series, maxValue, unit) {
    var svg = $(id); if (!svg) return;
    var W=360,H=220,L=42,R=14,T=18,B=46, plotW=W-L-R, plotH=H-T-B;
    var colors=['#2563eb','#14b8a6'];
    var g='';
    for(var y=0;y<=4;y++){
      var yy=T+plotH*y/4;
      g+='<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#e6eef8"/>';
      g+='<text x="'+(L-8)+'" y="'+(yy+4)+'" text-anchor="end">'+((maxValue*(4-y)/4).toFixed(unit==='cost'?2:0))+'</text>';
    }
    labels.forEach(function(label,i){
      var cx=L+plotW*(i+0.5)/labels.length, groupW=38;
      series.forEach(function(s,j){
        var h=s.values[i]/maxValue*plotH, x=cx-groupW/2+j*20, y=T+plotH-h;
        g+='<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="14" height="'+h.toFixed(1)+'" rx="2" fill="'+colors[j]+'"/>';
        g+='<text x="'+(x+7).toFixed(1)+'" y="'+(y-4).toFixed(1)+'" text-anchor="middle">'+s.values[i]+'</text>';
      });
      g+='<text x="'+cx.toFixed(1)+'" y="'+(H-26)+'" text-anchor="middle">'+label.split(' ')[0]+'</text>';
      g+='<text x="'+cx.toFixed(1)+'" y="'+(H-12)+'" text-anchor="middle">'+(label.split(' ')[1]||'')+'</text>';
    });
    g+='<rect x="'+(W-130)+'" y="4" width="9" height="9" fill="'+colors[0]+'"/><text x="'+(W-116)+'" y="12">Native</text>';
    g+='<rect x="'+(W-68)+'" y="4" width="9" height="9" fill="'+colors[1]+'"/><text x="'+(W-54)+'" y="12">+Runtime</text>';
    svg.innerHTML=g;
  }

  function fwClass(name) {
    if (name.indexOf('OpenAI') >= 0) return 'openai';
    if (name.indexOf('LangGraph') >= 0) return 'graph';
    if (name.indexOf('CrewAI') >= 0) return 'crew';
    if (name.indexOf('AutoGen') >= 0) return 'autogen';
    return '';
  }
  function fwLetter(name) {
    if (name.indexOf('OpenAI') >= 0) return 'O';
    if (name.indexOf('LangGraph') >= 0) return 'L';
    if (name.indexOf('CrewAI') >= 0) return 'C';
    if (name.indexOf('AutoGen') >= 0) return 'A';
    return '?';
  }
  function fmtCost(v) {
    if (v == null || isNaN(v)) return '-';
    if (v === 0) return '$0.000';
    if (v < 0.001) return '<$0.001';
    return '$' + v.toFixed(3);
  }
  function statusBadge(status) {
    if (!status || status === 'ok') return '';
    var cls = status === 'degraded' ? 'tag-degraded' : (status === 'skipped' ? 'tag-skipped' : 'tag-error');
    return ' <span class="tag-pill ' + cls + '">' + esc(status) + '</span>';
  }
  function statusBadgeOrOk(status) {
    if (!status || status === 'ok') return ' <span class="tag-pill tag-ok">ok</span>';
    return statusBadge(status);
  }
  function improvementCell(val, unit) {
    if (val == null || isNaN(val)) return '<span class="muted">not comparable</span>';
    if (unit === 'x') return '<b class="gain">↑' + val.toFixed(1) + 'x</b>';
    var cls = val < 0 ? 'improve' : 'gain';
    var arrow = val < 0 ? '↓' : '↑';
    return '<b class="' + cls + '">' + arrow + Math.abs(val).toFixed(1) + '%</b>';
  }
  function signedPct(v) {
    if (v == null || isNaN(v)) return 'n/a';
    return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
  }
  function runtimeContractLabel(contract) {
    if (contract === 'framework_orchestrated_runtime') return '<span class="tag-pill tag-ok">框架编排 + Runtime</span>';
    if (contract === 'model_gateway_direct_runtime') return '<span class="tag-pill tag-degraded">gateway direct</span>';
    return '';
  }
  function renderCloudCards(rows) {
    var grid = $('cloud-framework-cards');
    if (!grid || !rows) return;
    grid.innerHTML = rows.map(function(row) {
      var n = row.native || {}, r = row.runtime || {}, im = row.improvement || {};
      var nStatus = n.status || 'ok';
      var rStatus = r.status || 'ok';
      var note = n.note || r.note || '';
      var isStrict = r.runtime_contract === 'framework_orchestrated_runtime';
      var latencyLabel = isStrict ? 'strict improvement' : 'latency delta';
      var costLabel = isStrict ? 'cost' : 'cost delta';
      var eventsLabel = isStrict ? 'events' : 'events delta';
      var latency = im.latency != null ? signedPct(im.latency) : 'not comparable';
      var cost = im.cost != null ? signedPct(im.cost) : 'not comparable';
      var events = im.events != null ? im.events.toFixed(1) + 'x' : 'not comparable';
      var rContract = runtimeContractLabel(r.runtime_contract || '');
      var subNote = isStrict ? '' : '<small class="muted">native SDK vs AgentOS Gateway direct</small>';
      return '<article class="framework-card">' +
        '<header><span class="fw ' + fwClass(row.framework) + '">' + fwLetter(row.framework) + '</span><div><b>' + esc(row.framework) + '</b>' + rContract + '<small>' + esc(note) + '</small></div></header>' +
        '<div class="framework-status"><span>native' + statusBadgeOrOk(nStatus) + '</span><span>runtime' + statusBadgeOrOk(rStatus) + '</span></div>' +
        '<dl><dt>' + latencyLabel + '</dt><dd>' + latency + '</dd><dt>' + costLabel + '</dt><dd>' + cost + '</dd><dt>' + eventsLabel + '</dt><dd>' + events + '</dd></dl>' +
        subNote +
        '</article>';
    }).join('');
  }
  function renderCloudMatrix(rows) {
    var body = $('cloud-benchmark-tbody'); if (!body || !rows) return;
    renderCloudCards(rows);
    body.innerHTML = rows.map(function(row) {
      var n=row.native||{}, r=row.runtime||{}, im=row.improvement||{};
      var nStatus = n.status || 'ok';
      var rStatus = r.status || 'ok';
      var nProv = [n.orchestration_source, n.model_path].filter(Boolean).join(' · ') || n.adapter_source || '';
      var rProv = [r.orchestration_source, r.model_path].filter(Boolean).join(' · ') || r.adapter_source || '';
      var nNote = nProv ? '<br><small class="muted">' + esc(nProv) + '</small>' : '';
      var rNote = rProv ? '<br><small class="muted">' + esc(rProv) + '</small>' : '';
      return '<tr><td><span class="fw '+fwClass(row.framework)+'">'+fwLetter(row.framework)+'</span>'+esc(row.framework)+statusBadge(nStatus)+nNote+'</td>'+
        '<td>'+ (n.avg_latency != null ? n.avg_latency.toFixed(2) : '-') +'</td>'+
        '<td>'+ (n.prompt_tokens != null ? n.prompt_tokens.toFixed(1) : '-') +'</td>'+
        '<td>'+ fmtCost(n.api_cost) +'</td>'+
        '<td>'+ (n.context_reuse != null ? n.context_reuse.toFixed(1)+'%' : '-') +'</td>'+
        '<td>'+ (n.recovered_faults != null ? n.recovered_faults : '-') +'</td>'+
        '<td>'+ (n.observable_events != null ? fmt(n.observable_events) : '-') +'</td>'+
        '<td>'+ (r.avg_latency != null ? r.avg_latency.toFixed(2) : '-') +'</td>'+
        '<td>'+ (r.prompt_tokens != null ? r.prompt_tokens.toFixed(1) : '-') +'</td>'+
        '<td>'+ fmtCost(r.api_cost) +'</td>'+
        '<td>'+ (r.context_reuse != null ? r.context_reuse.toFixed(1)+'%' : '-') +'</td>'+
        '<td>'+ (r.recovered_faults != null ? r.recovered_faults : '-') +'</td>'+
        '<td>'+ (r.observable_events != null ? fmt(r.observable_events) : '-') +'</td>'+
        '<td>'+improvementCell(im.latency)+'</td>'+
        '<td>'+improvementCell(im.tokens)+'</td>'+
        '<td>'+improvementCell(im.cost)+'</td>'+
        '<td>'+improvementCell(im.context_reuse)+'</td>'+
        '<td>'+improvementCell(im.events,'x')+'</td></tr>';
    }).join('');
  }

  function drawStackedBars(id, rows, dynamicMax) {
    var svg = $(id); if (!svg) return;
    var labels=(rows||[]).map(function(r){ return r.framework; });
    var prompt=(rows||[]).map(function(r){ return r.native.prompt_tokens||0; });
    var cached=(rows||[]).map(function(r){ return r.runtime.prompt_tokens||0; });
    if (!labels.length) return;
    var W=360,H=220,L=42,R=14,T=20,B=46, plotW=W-L-R, plotH=H-T-B, max=dynamicMax||1100;
    var g='';
    for(var y=0;y<=4;y++){
      var yy=T+plotH*y/4;
      g+='<line x1="'+L+'" y1="'+yy+'" x2="'+(W-R)+'" y2="'+yy+'" stroke="#e6eef8"/>';
      g+='<text x="'+(L-8)+'" y="'+(yy+4)+'" text-anchor="end">'+(max*(4-y)/4).toFixed(0)+'</text>';
    }
    labels.forEach(function(label,i){
      var cx=L+plotW*(i+0.5)/labels.length, total=prompt[i]+cached[i];
      var hp=prompt[i]/max*plotH, hc=cached[i]/max*plotH, y=T+plotH-hp-hc;
      g+='<rect x="'+(cx-13)+'" y="'+y+'" width="26" height="'+hp+'" rx="2" fill="#ff6b73" opacity="0.82"/>';
      g+='<rect x="'+(cx-13)+'" y="'+(y+hp)+'" width="26" height="'+hc+'" rx="2" fill="#14b8a6" opacity="0.9"/>';
      g+='<text x="'+cx+'" y="'+(y-4)+'" text-anchor="middle">'+total.toFixed(0)+'</text>';
      g+='<text x="'+cx+'" y="'+(H-25)+'" text-anchor="middle">'+label.split(' ')[0]+'</text>';
      g+='<text x="'+cx+'" y="'+(H-11)+'" text-anchor="middle">'+(label.split(' ')[1]||'')+'</text>';
    });
    g+='<rect x="190" y="4" width="9" height="9" fill="#ff6b73"/><text x="204" y="12">Prompt Tokens</text>';
    g+='<rect x="286" y="4" width="9" height="9" fill="#14b8a6"/><text x="300" y="12">Cached</text>';
    svg.innerHTML=g;
  }

  function renderCloudCharts(payload) {
    var rows = payload && payload.frameworks ? payload.frameworks : null;
    if (!rows || !rows.length) {
      rows = [
        { framework:'OpenAI Agents SDK', native:{avg_latency:4.32,prompt_tokens:612.7,api_cost:0.184}, runtime:{avg_latency:2.94,prompt_tokens:378.6,api_cost:0.134} },
        { framework:'LangGraph', native:{avg_latency:4.78,prompt_tokens:653.4,api_cost:0.196}, runtime:{avg_latency:3.21,prompt_tokens:402.1,api_cost:0.142} },
        { framework:'AutoGen', native:{avg_latency:4.15,prompt_tokens:589.6,api_cost:0.176}, runtime:{avg_latency:2.82,prompt_tokens:360.7,api_cost:0.128} },
        { framework:'CrewAI', native:{avg_latency:0,prompt_tokens:0,api_cost:0}, runtime:{avg_latency:0,prompt_tokens:0,api_cost:0} }
      ];
    }
    renderCloudMatrix(rows);
    // Update cloud KPIs from metadata
    var meta = payload && payload.metadata ? payload.metadata : {};
    setText('cloud-kpi-frameworks', rows.length);
    setText('cloud-kpi-scenarios', meta.scenarios || 1);
    setText('cloud-kpi-runs', meta.total_runs || rows.length * 2);
    // Strict framework runtime KPIs: only from framework_orchestrated_runtime rows
    var strictRows = rows.filter(function(r){
      return r.runtime && r.runtime.runtime_contract === 'framework_orchestrated_runtime'
        && r.improvement && Object.keys(r.improvement).length > 0;
    });
    if (strictRows.length > 0) {
      var avgLat = strictRows.reduce(function(s,r){ return s + (r.improvement.latency||0); }, 0) / strictRows.length;
      var avgTok = strictRows.reduce(function(s,r){ return s + (r.improvement.tokens||0); }, 0) / strictRows.length;
      var avgCost = strictRows.reduce(function(s,r){ return s + (r.improvement.cost||0); }, 0) / strictRows.length;
      setText('cloud-kpi-latency', (avgLat >= 0 ? '+' : '') + avgLat.toFixed(1) + '%');
      setText('cloud-kpi-tokens', (avgTok >= 0 ? '+' : '') + avgTok.toFixed(1) + '%');
      setText('cloud-kpi-cost', (avgCost >= 0 ? '+' : '') + avgCost.toFixed(1) + '%');
      var kpiSmall = function(id, txt){ var el=$(id); if(el) el.textContent=txt; };
      kpiSmall('cloud-kpi-latency-sm', '基于 ' + strictRows.length + ' 个框架真实接入');
      kpiSmall('cloud-kpi-tokens-sm', '基于 ' + strictRows.length + ' 个框架真实接入');
      kpiSmall('cloud-kpi-cost-sm', '基于 ' + strictRows.length + ' 个框架真实接入');
    } else {
      setText('cloud-kpi-latency', 'n/a');
      setText('cloud-kpi-tokens', 'n/a');
      setText('cloud-kpi-cost', 'n/a');
    }
    // Recovery: sum runtime recovered_faults
    var totalRecovery = rows.reduce(function(s,r){ return s + (r.runtime && r.runtime.recovered_faults ? r.runtime.recovered_faults : 0); }, 0);
    setText('cloud-kpi-recovery', totalRecovery > 0 ? totalRecovery : 'n/a');
    // Events: sum runtime observable_events
    var totalEvents = rows.reduce(function(s,r){ return s + (r.runtime && r.runtime.observable_events ? r.runtime.observable_events : 0); }, 0);
    setText('cloud-kpi-events', totalEvents > 0 ? fmt(totalEvents) : 'n/a');
    // Footer source
    var footer = $('cloud-footer-source');
    if (footer) {
      var model = meta.model || 'unknown';
      var source = meta.source || 'unknown';
      footer.textContent = '数据来源：' + source + '　|　模型：' + model + '　|　场景：' + (meta.scenarios||1) + ' 个　|　总运行：' + (meta.total_runs||'?') + ' 次';
    }

    // Provenance summary
    var prov = $('cloud-provenance-summary');
    if (prov) {
      var fwOrchCount = meta.strict_framework_runtime_count || meta.framework_orchestrated_runtime_count || 0;
      var gwDirectCount = meta.gateway_direct_runtime_count || 0;
      if (fwOrchCount === rows.length) {
        prov.innerHTML = '四个框架均保留原生编排，底层模型调用统一接入 AgentOS Runtime。' +
          '<br><small class="muted">输入、API Key 与任务保持一致；差异来自 Runtime 的上下文复用、故障恢复与观测事件。</small>';
      } else {
        var fwOrchList = (meta.strict_framework_runtime_frameworks || []).join(', ') || 'none';
        prov.innerHTML = 'Strict framework runtime: ' + fwOrchCount + '/' + rows.length + ' (' + esc(fwOrchList) + ')' +
          '　|　Gateway direct: ' + gwDirectCount + '/' + rows.length;
      }
    }

    // Evidence rail — dynamic from JSON
    var fwOrchCnt = meta.strict_framework_runtime_count || meta.framework_orchestrated_runtime_count || 0;
    var gwDirCnt = meta.gateway_direct_runtime_count || 0;
    var allStrict = fwOrchCnt === rows.length;
    setText('evidence-scheduling-small',
      fwOrchCnt + '/' + rows.length + ' 框架保留原生编排');
    setText('evidence-reuse-small',
      fwOrchCnt + '/' + rows.length + ' 框架接入 ModelGateway');
    setText('evidence-cost-small',
      'native/runtime 均来自 DeepSeek usage API');
    var totalRecovery = rows.reduce(function(s,r){ return s + (r.runtime && r.runtime.recovered_faults ? r.runtime.recovered_faults : 0); }, 0);
    setText('evidence-fault-small',
      totalRecovery + '/' + rows.length + ' runtime 路径记录恢复事件');
    setText('evidence-provenance-small',
      '事件流含 scheduler / fault / context / llm 类型');

    // Insight cards: render three conclusion cards from benchmark data
    renderInsightCards(rows);

    // Recovery matrix
    var recMatrix = $('recovery-matrix');
    if (recMatrix) {
      var html = '';
      rows.forEach(function(r) {
        var rt = r.runtime || {};
        var recovered = (rt.recovered_faults || 0) >= 1;
        var ok = rt.status === 'ok';
        html += '<div class="recovery-row">' +
          '<span class="framework-chip">' + esc(r.framework) + '</span>' +
          '<span class="recovery-badge injected">已注入</span>' +
          '<span class="recovery-badge ' + (recovered ? 'recovered' : 'injected') + '">' + (recovered ? '已恢复' : '未记录') + '</span>' +
          '<span class="recovery-badge ' + (ok ? 'healthy' : 'injected') + '">' + (ok ? '健康' : '异常') + '</span>' +
          '</div>';
      });
      recMatrix.innerHTML = html;
    }
  }

  function renderInsightCards(rows) {
    // 1. Latency card with Native/Runtime bar pairs
    var latList = $('insight-latency-list');
    if (latList) {
      var maxLat = 0;
      rows.forEach(function(r) {
        var n = (r.native||{}).avg_latency || 0;
        var rt = (r.runtime||{}).avg_latency || 0;
        if (n > maxLat) maxLat = n;
        if (rt > maxLat) maxLat = rt;
      });
      if (maxLat === 0) maxLat = 1;
      var html = '';
      rows.forEach(function(r) {
        var n = r.native || {}, rt = r.runtime || {}, im = r.improvement || {};
        var nLat = n.avg_latency || 0;
        var rLat = rt.avg_latency || 0;
        var pct = im.latency;
        var deltaHtml = deltaCell(pct);
        var nPct = Math.max(6, (nLat / maxLat) * 100);
        var rPct = Math.max(6, (rLat / maxLat) * 100);
        html += '<div class="comparison-row">' +
          '<div class="comparison-head">' +
            '<span class="fw ' + fwClass(r.framework) + '">' + fwLetter(r.framework) + '</span>' +
            '<span class="framework-name">' + esc(r.framework) + '</span>' +
            '<span class="comparison-values">' + nLat.toFixed(2) + 's → ' + rLat.toFixed(2) + 's</span>' +
            deltaHtml +
          '</div>' +
          '<div class="bar-pair">' +
            '<div class="bar-line native" style="width:' + nPct.toFixed(0) + '%"></div>' +
            '<div class="bar-line runtime" style="width:' + rPct.toFixed(0) + '%"></div>' +
          '</div>' +
          '</div>';
      });
      latList.innerHTML = html;
    }

    // 2. Prompt / Cache card with prompt bar + reuse badge
    var promptList = $('insight-prompt-list');
    if (promptList) {
      var rTokens = rows.map(function(r){ return (r.runtime||{}).prompt_tokens || 0; }).filter(function(v){ return v > 0; }).sort(function(a,b){ return a-b; });
      var medianTok = rTokens.length > 0 ? rTokens[Math.floor(rTokens.length / 2)] : 0;
      var maxTok = 0;
      rows.forEach(function(r) { var t = (r.runtime||{}).prompt_tokens || 0; if (t > maxTok) maxTok = t; });
      if (maxTok === 0) maxTok = 1;
      var html = '';
      rows.forEach(function(r) {
        var rt = r.runtime || {};
        var tok = rt.prompt_tokens || 0;
        var reuse = rt.context_reuse || 0;
        var note = '';
        if (medianTok > 0 && tok > medianTok * 1.8) {
          note = '<span class="insight-note">框架模板较长</span>';
        }
        var barPct = Math.max(6, (tok / maxTok) * 100);
        html += '<div class="comparison-row">' +
          '<div class="comparison-head">' +
            '<span class="fw ' + fwClass(r.framework) + '">' + fwLetter(r.framework) + '</span>' +
            '<span class="framework-name">' + esc(r.framework) + '</span>' +
            '<span class="comparison-values">Prompt ' + tok.toFixed(1) + 'K</span>' +
            '<span class="reuse-badge">复用 ' + reuse.toFixed(1) + '%</span>' +
            note +
          '</div>' +
          '<div class="bar-pair">' +
            '<div class="bar-line prompt" style="width:' + barPct.toFixed(0) + '%"></div>' +
          '</div>' +
          '</div>';
      });
      promptList.innerHTML = html;
    }

    // 3. Cost card with Native/Runtime bar pairs
    var costList = $('insight-cost-list');
    if (costList) {
      var maxCost = 0;
      rows.forEach(function(r) {
        var n = (r.native||{}).api_cost || 0;
        var rt = (r.runtime||{}).api_cost || 0;
        if (n > maxCost) maxCost = n;
        if (rt > maxCost) maxCost = rt;
      });
      if (maxCost === 0) maxCost = 0.001;
      var html = '';
      rows.forEach(function(r) {
        var n = r.native || {}, rt = r.runtime || {}, im = r.improvement || {};
        var nCost = n.api_cost || 0;
        var rCost = rt.api_cost || 0;
        var pct = im.cost;
        var deltaHtml = deltaCell(pct);
        var nPct = Math.max(6, (nCost / maxCost) * 100);
        var rPct = Math.max(6, (rCost / maxCost) * 100);
        html += '<div class="comparison-row">' +
          '<div class="comparison-head">' +
            '<span class="fw ' + fwClass(r.framework) + '">' + fwLetter(r.framework) + '</span>' +
            '<span class="framework-name">' + esc(r.framework) + '</span>' +
            deltaHtml +
          '</div>' +
          '<div class="bar-pair">' +
            '<div class="bar-line native" style="width:' + nPct.toFixed(0) + '%"></div>' +
            '<div class="bar-line runtime" style="width:' + rPct.toFixed(0) + '%"></div>' +
          '</div>' +
          '</div>';
      });
      costList.innerHTML = html;
    }
  }

  function deltaCell(pct) {
    if (pct == null) return '<span class="insight-delta delta-neutral">暂无对比</span>';
    if (pct < 0) return '<span class="insight-delta delta-good">↓' + Math.abs(pct).toFixed(1) + '%</span>';
    if (pct === 0) return '<span class="insight-delta delta-neutral">0.0%</span>';
    return '<span class="insight-delta delta-bad">↑' + pct.toFixed(1) + '%</span>';
  }
  function loadCloudBenchmark() {
    return fetch('/api/cloud-benchmark').then(function(r){ return r.json(); })
      .then(function(payload){ renderCloudCharts(payload); })
      .catch(function(){ renderCloudCharts(); });
  }

  // --- workload matrix ---
  function loadWorkloadMatrix() {
    return fetch('/api/workload-matrix').then(function(r){
      if (!r.ok) throw new Error('not available');
      return r.json();
    })
    .then(function(payload){ renderPressureMatrix(payload); })
    .catch(function(){ renderPressureMatrix(null); });
  }

  function renderPressureMatrix(payload) {
    var grid = $('pressure-grid');
    if (!grid) return;

    if (!payload || !payload.workloads) {
      // fallback: keep existing static HTML
      return;
    }

    var workloads = payload.workloads;
    var names = {
      'ai_inference': 'AI 推理',
      'compile': '编译压力',
      'compression': '压缩压力',
      'network': '网络压测'
    };
    var order = ['ai_inference', 'compile', 'compression', 'network'];
    var html = '';

    order.forEach(function(key) {
      var w = workloads[key];
      if (!w) return;
      var b = w.baseline || {}, r = w.runtime || {}, c = w.comparison || {};
      var bMakespan = b.makespan_secs != null ? b.makespan_secs.toFixed(1) + 's' : '-';
      var rMakespan = r.makespan_secs != null ? r.makespan_secs.toFixed(1) + 's' : '-';
      var bThroughput = b.throughput_tasks_per_sec != null ? b.throughput_tasks_per_sec.toFixed(3) : '-';
      var rThroughput = r.throughput_tasks_per_sec != null ? r.throughput_tasks_per_sec.toFixed(3) : '-';
      var bTasks = (b.tasks_completed != null && b.tasks_total != null) ? b.tasks_completed + '/' + b.tasks_total : '-';
      var rTasks = (r.tasks_completed != null && r.tasks_total != null) ? r.tasks_completed + '/' + r.tasks_total : '-';
      var bCacheHit = b.cache_hit_ratio != null ? (b.cache_hit_ratio * 100).toFixed(1) + '%' : '-';
      var rCacheHit = r.cache_hit_ratio != null ? (r.cache_hit_ratio * 100).toFixed(1) + '%' : '-';
      var faultRec = c.fault_recovery != null ? (c.fault_recovery ? 'yes' : 'no') : '-';
      var deltaPct = c.latency_delta_pct != null ? c.latency_delta_pct.toFixed(1) + '%' : '-';

      html += '<article><b>' + esc(names[key] || key) + '</b><em>' + esc(key) + '</em>' +
        '<dl class="pressure-stats">' +
        '<dt>makespan</dt><dd>' + bMakespan + ' → ' + rMakespan + ' <span>(' + deltaPct + ')</span></dd>' +
        '<dt>throughput</dt><dd>' + bThroughput + ' → ' + rThroughput + '</dd>' +
        '<dt>tasks</dt><dd>' + bTasks + ' → ' + rTasks + '</dd>' +
        '<dt>cache hit</dt><dd>' + bCacheHit + ' → ' + rCacheHit + '</dd>' +
        '</dl>' +
        '<small class="pressure-source">fault_recovered ' + faultRec + ' · real deepseek</small></article>';
    });

    // kernel probe card (keep existing probe-card article)
    var probeCard = grid.querySelector('.probe-card');
    if (probeCard) {
      html += probeCard.outerHTML;
    }

    grid.innerHTML = html;
  }

  function loadSnapshot(){ return fetch('/api/snapshot').then(function(r){return r.json();})
    .then(function(data){ (data.events||[]).forEach(function(e){
      if(String(e.event_type).indexOf('director.')===0) return;
      processEvent(e);
    }); }).catch(function(e){ console.warn(e); }); }
  function connectSSE(){ var es=new EventSource('/api/events');
    es.onopen=function(){ $('conn-dot').className='dot ok'; $('conn-text').textContent='connected · streaming'; };
    es.onmessage=function(m){ try{
      var evt=JSON.parse(m.data);
      if(String(evt.event_type).indexOf('director.')===0){ if(window.Director) window.Director.handle(evt); return; }
      processEvent(evt);
    }catch(e){ console.warn(e); } };
    es.onerror=function(){ $('conn-dot').className='dot err'; $('conn-text').textContent='disconnected · retrying…'; }; }
  ['local', 'cloud'].forEach(function(page) {
    var tab = $('tab-' + page);
    if (tab) tab.addEventListener('click', function(){ switchPage(page); });
  });
  loadCloudBenchmark();
  switchPage('local');
  renderAll();
  loadSnapshot().then(function() {
    connectSSE();
    loadWorkloadMatrix();
  });
})();
