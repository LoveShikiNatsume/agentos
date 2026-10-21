/* The workbench reads the same run events and summary used by agentctl. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const API = '/api/workbench';
  const paths = {
    workbench:'M8 5H4a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h4m8-12h4a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1h-4M8 3h8v18H8z',
    activity:'M2 12h4l3-8 5 16 3-8h5', plus:'M12 5v14M5 12h14',
    external:'M14 3h7v7m0-7L10 14M10 5H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5',
    refresh:'M20 7a8 8 0 1 0 0 10M20 3v5h-5',
    branch:'M7 7v10m0-5h7a4 4 0 0 0 4-4V7M7 3a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm11 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM7 17a2 2 0 1 0 0 4 2 2 0 0 0 0-4Z',
    spark:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z',
    file:'M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 12h8M8 16h6',
    code:'m8 6-6 6 6 6m8-12 6 6-6 6m-3-14-2 16',
    folder:'M3 6h7l2 2h9v11H3V6Z',
    layers:'m12 3 9 5-9 5-9-5 9-5ZM3 12l9 5 9-5M3 16l9 5 9-5',
    cpu:'M6 6h12v12H6zM9 9h6v6H9zM9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4',
    settings:'M4 7h16M4 17h16M8 4v6m8 4v6',
    stop:'M6 6h12v12H6z', arrow:'M5 12h14m-6-6 6 6-6 6',
    down:'M12 4v16m-6-6 6 6 6-6', close:'m6 6 12 12M6 18 18 6',
    robot:'M6 8h12a2 2 0 0 1 2 2v8H4v-8a2 2 0 0 1 2-2Zm6 0V4M10 4h4M8 12h.01M16 12h.01M8 16h8M2 11v4m20-4v4',
    user:'M8 7a4 4 0 1 0 8 0 4 4 0 0 0-8 0ZM4 21v-3a8 6 0 0 1 16 0v3',
    check:'m5 12 4 4L19 6', message:'M4 4h16v12H9l-5 4V4Z',
    tool:'m14 6 4 4-9 10-4-4 9-10Zm0 0a5 5 0 0 1 7-3l-3 3 1 2 3-2a5 5 0 0 1-4 4',
    clock:'M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm0 3v5l3 2',
    shield:'m12 3 8 3v6c0 4-4 7-8 9-4-2-8-5-8-9V6l8-3Zm-4 9 3 3 5-6',
    warning:'m12 3 10 18H2L12 3Zm0 6v5m0 3h.01',
  };
  const icon = name => `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name] || paths.message}"/></svg>`;
  function hydrateIcons(root=document) { root.querySelectorAll('[data-icon]').forEach(el => { el.innerHTML = icon(el.dataset.icon); }); }
  function el(tag, cls, text) { const n=document.createElement(tag); if(cls)n.className=cls; if(text!==undefined)n.textContent=String(text); return n; }
  function glyph(name, cls='') { const n=el('span',cls); n.innerHTML=icon(name); return n; }
  const labels={queued:'准备中',running:'运行中',completed:'已完成',failed:'未完整结束',cancelled:'已停止',cancelling:'正在停止',pending:'等待依赖',ready:'就绪',skipped:'未执行'};
  const activeState = state => ['queued','running','cancelling'].includes(state);
  const number = n => typeof n==='number' && Number.isFinite(n);
  const fmt = n => number(n) ? new Intl.NumberFormat('en-US').format(n) : '—';
  const short = (s,n=70) => String(s||'').length>n ? String(s).slice(0,n)+'…' : String(s||'');
  const epoch = value => typeof value==='number' ? value*1000 : Date.parse(value||'');
  const duration = value => number(value) ? value.toFixed(1)+' s' : '—';
  function cleanJSON(value) { return JSON.stringify(value,(key,v)=>key==='reasoning_content'?undefined:v,2); }
  const S={config:null,jobs:[],job:null,id:null,events:[],cursor:0,tasks:new Map(),agents:new Map(),models:new Map(),entries:[],tools:new Map(),refs:new Map(),commands:[],messages:0,published:0,selected:null,tab:'process',view:'workbench',generation:0,follow:true,open:new Set(),artifactFiles:[],artifactAnswers:{},artifactsLoaded:false,online:false,polling:false,fetchingJob:false,submitting:false,lastJobs:0,renderKey:''};
  const welcome=$('process-panel').cloneNode(true);
  let toastTimer;
  function toast(message, success=false) { $('toast').textContent=message; $('toast').classList.toggle('success',success); $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>{$('toast').hidden=true;},7000); }
  function connection(ok) { S.online=ok; const n=$('connection'); n.className='connection '+(ok?'online':'offline'); n.replaceChildren(el('i'),document.createTextNode(ok?'服务已连接':'连接中断')); }
  async function api(path, options={}) {
    const controller=new AbortController(); const timer=setTimeout(()=>controller.abort(),15000);
    try {
      const headers={...(options.body?{'Content-Type':'application/json'}:{}),...options.headers};
      const r=await fetch(API+path,{...options,headers,signal:controller.signal});
      let data; try { data=await r.json(); } catch { throw new Error('本地服务返回了无法读取的响应。'); }
      if(!r.ok) throw new Error(typeof data.error==='string'?data.error:data.message||data.error?.message||`请求失败（${r.status}）`);
      connection(true); return data;
    } catch(e) { if(e.name==='AbortError'||e instanceof TypeError) connection(false); throw e; }
    finally { clearTimeout(timer); }
  }
  function ensureTask(id, data={}) {
    if(!id)return null;
    if(!S.tasks.has(id))S.tasks.set(id,{id,name:id==='planner'?'任务规划':id,goal:'',dependencies:[],status:'pending',role:id==='planner'?'planner':'worker'});
    const t=S.tasks.get(id); if(data.name||data.goal){t.goal=data.name||data.goal; t.name=id==='planner'?'任务规划':t.goal;}
    if(Array.isArray(data.dependencies))t.dependencies=data.dependencies;
    if(data.agent_role||data.role)t.role=data.agent_role||data.role;
    return t;
  }
  function taskTitle(t) { return t.id==='planner'?'任务规划':/[\u3400-\u9fff]/.test(t.name)?t.name.split(/[。\n]/)[0]:t.id.replaceAll('_',' ').replaceAll('-',' '); }
  function owner(id) { const t=S.tasks.get(id); return id==='planner'?'规划 Agent':t?short(taskTitle(t),30):id||'Runtime'; }
  function stateDot(state){return el('i','status-dot '+state);}
  function badge(state){return el('span','state-badge '+state,labels[state]||state||'等待任务');}
  function addEntry(e, kind, title, text, extra={}) { S.entries.push({key:'e-'+S.events.length,event:e,kind,title,text,...extra}); }
  function ingest(e) {
    S.events.push(e); const d=e.data||{}, type=e.event_type, tid=e.task_id;
    if(e.agent_id && tid && !S.agents.has(tid))S.agents.set(tid,{id:e.agent_id,taskId:tid});
    if(type==='agent.created'){ const t=ensureTask(tid,d); t.agentId=e.agent_id; }
    if(type==='task.dynamic_batch_submitted') for(const task of d.tasks||[])ensureTask(task.task_id,task);
    if(type==='harness.plan_created')addEntry(e,'plan','任务规划',`已生成 ${(d.tasks||[]).length} 项任务，按依赖关系交接执行。`,{plan:d.tasks||[]});
    if(type==='execution.started')addEntry(e,'user','用户任务',d.task||S.job?.task||'');
    if(type==='task.started'){const t=ensureTask(tid);t.status='running';t.started=e.timestamp;addEntry(e,'start',tid==='planner'?'开始规划':'开始子任务',tid==='planner'?'分析任务并生成执行计划。':t.goal||tid);}
    if(type==='task.completed'){const t=ensureTask(tid);t.status='completed';t.finished=e.timestamp;addEntry(e,'complete',tid==='planner'?'规划完成':'子任务完成',tid==='planner'?'执行计划已交给后续 Agent。':d.result_preview||'执行结果已记录。');}
    if(type==='task.failed'){const t=ensureTask(tid);t.status='failed';t.finished=e.timestamp;addEntry(e,'error','子任务未完成',d.error||'请查看执行详情。');}
    if(type==='fault.propagated'){const t=ensureTask(tid);if(t&&t.status!=='completed')t.status='skipped';}
    if(type==='execution.model_started')S.models.set(`${d.call_id}:${d.attempt}`,{task:tid,state:'running',started:e.timestamp});
    if(type==='execution.model_completed'){const key=`${d.call_id}:${d.attempt}`;S.models.set(key,{...(S.models.get(key)||{}),task:tid,state:'completed',usage:d.usage,metrics:d.metrics});}
    if(type==='execution.model_failed'){const key=`${d.call_id}:${d.attempt}`;S.models.set(key,{...(S.models.get(key)||{}),task:tid,state:'failed'});addEntry(e,'error','模型调用异常',d.error||'调用没有正常完成。');}
    if(type==='execution.tool_started'){
      const key=`${tid}:${d.tool}`; const entry={key:'e-'+S.events.length,event:e,kind:'tool',title:d.tool,args:d.arguments||{},pending:true};S.entries.push(entry);
      if(!S.tools.has(key))S.tools.set(key,[]);S.tools.get(key).push(entry);
    }
    if(type==='execution.tool_finished'){
      const entry=(S.tools.get(`${tid}:${d.tool}`)||[]).find(x=>x.pending);
      if(entry){entry.pending=false;entry.output=d.output;entry.error=d.error;entry.ok=d.ok;}
    }
    if(type==='execution.command_result')S.commands.push({task:tid,...d,timestamp:e.timestamp});
    if(type==='context.assembled')for(const ref of d.shared_refs||[]){if(!S.refs.has(ref))S.refs.set(ref,{name:ref,consumers:new Set(),published:false});S.refs.get(ref).consumers.add(tid);}
    if(type==='context.result_published'){
      if(!S.refs.has(d.ref))S.refs.set(d.ref,{name:d.ref,consumers:new Set(),published:false});S.refs.get(d.ref).published=true;S.published++;
    }
    if(type==='message.reference_sent')S.messages++;
    if(type==='message.reference_received')addEntry(e,'handoff','收到上游结果',`${d.ref||'共享结果'} · 来自 ${d.sender||'上游 Agent'}`);
    if(type==='harness.protocol_error')addEntry(e,'notice','正在修正调用格式',d.error||'已将格式问题反馈给模型。');
    if(type==='execution.failed')addEntry(e,'error','本次执行未完整结束',d.error||'已有执行记录和产物会被保留。');
    if(type==='execution.completed')addEntry(e,d.workflow_status==='completed'?'complete':'notice',d.workflow_status==='completed'?'本次执行完成':'本次执行已结束','执行记录已保存，可查看任务产物与实际用量。');
  }
  function applySummary() { for(const [id,status] of Object.entries(S.job?.summary?.task_statuses||{}))ensureTask(id).status=status; }
  function resetData(){ S.events=[];S.cursor=0;S.tasks=new Map();S.agents=new Map();S.models=new Map();S.entries=[];S.tools=new Map();S.refs=new Map();S.commands=[];S.messages=0;S.published=0;S.selected=null;S.open=new Set();S.artifactFiles=[];S.artifactAnswers={};S.artifactsLoaded=false;S.renderKey='';S.follow=true; }
  async function selectJob(job) {
    ++S.generation;resetData();S.job=job;S.id=job?.id||null;
    if(job){$('task-input').value=job.task||'';$('cwd-input').value=job.cwd||$('cwd-input').value;if(job.config?.model)$('model-input').value=job.config.model;if(job.config?.base_url)$('base-url-input').value=job.config.base_url;history.replaceState(null,'','#job='+encodeURIComponent(job.id));}
    else {history.replaceState(null,'',location.pathname);$('task-input').value='';}
    setTab('process');renderAll();if(job)await pollSelected();else $('task-input').focus();
  }
  async function refreshJobs(){const result=await api('/jobs');S.jobs=result.jobs||[];S.lastJobs=Date.now();renderSidebar();updateControls();}
  async function pollSelected(){
    if(!S.id||S.fetchingJob)return;S.fetchingJob=true;
    try { const id=S.id,gen=S.generation;
    const job=await api('/jobs/'+encodeURIComponent(id));if(gen!==S.generation)return;S.job=job;
    let rounds=0;
    while(rounds++<8){
      const data=await api(`/jobs/${encodeURIComponent(id)}/events?after=${S.cursor}`);if(gen!==S.generation)return;
      const incoming=data.events||[];incoming.forEach(ingest);S.cursor=data.next_cursor??(S.cursor+incoming.length);
      if(data.run_id)S.job.run_id=data.run_id;
      if(incoming.length<500)break;
    }
    applySummary();
    if(!activeState(S.job.state)&&!S.artifactsLoaded){
      const data=await api(`/jobs/${encodeURIComponent(id)}/artifacts`);if(gen!==S.generation)return;
      S.artifactFiles=data.files||[];S.artifactAnswers=data.answers||S.job.summary?.answers||{};S.artifactsLoaded=true;
    }
    S.jobs=S.jobs.map(x=>x.id===id?S.job:x);renderAll();
    } finally { S.fetchingJob=false; }
  }
  async function tick(){
    if(S.polling)return;S.polling=true;
    try {if(Date.now()-S.lastJobs>3500)await refreshJobs();if(S.id&&(activeState(S.job?.state)||!S.artifactsLoaded))await pollSelected();}
    catch(e){ if(e.name!=='AbortError')console.warn('工作台连接:',e.message);connection(false); }
    finally {S.polling=false;setTimeout(tick,850);}
  }
  function renderSidebar(){
    $('agent-count').textContent=S.agents.size;
    const agents=document.createDocumentFragment();
    if(S.job){
      const all=el('button','agent-item'+(!S.selected?' selected':''));all.dataset.task='';all.append(glyph('layers','agent-glyph'));
      const copy=el('div','agent-copy');copy.append(el('span','agent-name','全部执行记录'));const meta=el('div','agent-meta');meta.append(el('span','',`${S.tasks.size} 项任务`),el('span','',labels[S.job.state]||''));copy.append(meta);all.append(copy);agents.append(all);
    }
    for(const [tid,a] of S.agents){
      const t=S.tasks.get(tid)||{name:tid,status:'pending'};const n=el('button','agent-item'+(S.selected===tid?' selected':''));n.dataset.task=tid;n.title=t.goal||t.name;
      n.append(glyph(tid==='planner'?'branch':'robot','agent-glyph'));const copy=el('div','agent-copy');copy.append(el('span','agent-name',tid==='planner'?'规划 Agent':taskTitle(t)));
      const meta=el('div','agent-meta');meta.append(el('span','agent-id',a.id));const state=el('span','mini-state');state.append(stateDot(t.status),document.createTextNode(labels[t.status]||t.status));meta.append(state);copy.append(meta);n.append(copy);agents.append(n);
    }
    if(!S.job)agents.append(el('p','sidebar-empty','提交后显示实际运行的 Agent。'));
    $('agent-list').replaceChildren(agents);
    const jobs=document.createDocumentFragment();
    for(const job of S.jobs.slice(0,12)){
      const n=el('button','job-item'+(job.id===S.id?' selected':''));n.dataset.job=job.id;n.title=job.task||job.id;n.append(stateDot(job.state),el('span','',short((job.task||'未命名任务').split('\n')[0],60)));
      const date=new Date(epoch(job.created_at));n.append(el('small','job-time',Number.isNaN(date.getTime())?'':date.toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false})));jobs.append(n);
    }
    if(!S.jobs.length)jobs.append(el('p','sidebar-empty','还没有运行记录。'));
    $('job-list').replaceChildren(jobs);
  }
  function setSelected(id){S.selected=id||null;S.renderKey='';setTab('process');renderAll();}
  function renderDAG(){
    const tasks=[...S.tasks.values()];$('dag-summary').textContent=tasks.length?`${tasks.length} 项任务 · ${S.agents.size} 个 Agent · 完成 ${tasks.filter(t=>t.status==='completed').length}/${tasks.length}`:'提交任务后生成';
    if(!tasks.length){const empty=el('div','dag-empty');empty.append(glyph('branch'),el('span','',S.job?'等待规划生成真实任务依赖图':'真实任务的依赖与执行状态，将显示在这里'));$('dag').replaceChildren(empty);return;}
    const levels=new Map();const depth=(id,seen=new Set())=>{if(levels.has(id))return levels.get(id);if(seen.has(id))return 0;const t=S.tasks.get(id);if(!t)return 0;seen.add(id);const n=t.dependencies.length?1+Math.max(...t.dependencies.map(d=>depth(d,new Set(seen)))):0;levels.set(id,n);return n;};tasks.forEach(t=>depth(t.id));
    const groups=new Map();for(const t of tasks){const level=levels.get(t.id);if(!groups.has(level))groups.set(level,[]);groups.get(level).push(t);}
    const layout=el('div','dag-layout');const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('dag-wires');svg.setAttribute('aria-hidden','true');layout.append(svg);
    for(const level of [...groups.keys()].sort((a,b)=>a-b)){
      const column=el('div','dag-column');for(const t of groups.get(level)){
        const n=el('button','dag-node '+t.status+(S.selected===t.id?' selected':''));n.dataset.task=t.id;n.title=t.goal||t.name;n.setAttribute('aria-label',`${t.name}，${labels[t.status]||t.status}`);n.append(el('span','dag-name',taskTitle(t)));const st=el('span','dag-state');st.append(stateDot(t.status),document.createTextNode(labels[t.status]||t.status));n.append(st);column.append(n);
      }layout.append(column);
    }
    const oldScroll=$('dag').scrollLeft;$('dag').replaceChildren(layout);$('dag').scrollLeft=oldScroll;requestAnimationFrame(drawWires);
  }
  function drawWires(){
    const layout=$('dag').querySelector('.dag-layout'),svg=layout?.querySelector('svg');if(!layout||!svg)return;
    const ns='http://www.w3.org/2000/svg',r=layout.getBoundingClientRect();svg.replaceChildren();svg.setAttribute('width',layout.scrollWidth);svg.setAttribute('height',layout.scrollHeight);
    const defs=document.createElementNS(ns,'defs'),marker=document.createElementNS(ns,'marker');marker.id='dag-arrow';marker.setAttribute('markerWidth','5');marker.setAttribute('markerHeight','6');marker.setAttribute('refX','4');marker.setAttribute('refY','3');marker.setAttribute('orient','auto');const tip=document.createElementNS(ns,'path');tip.setAttribute('d','M0 0 L5 3 L0 6');tip.setAttribute('fill','#b6c4d8');marker.append(tip);defs.append(marker);svg.append(defs);
    const nodes=new Map([...layout.querySelectorAll('[data-task]')].map(n=>[n.dataset.task,n]));
    for(const t of S.tasks.values())for(const dep of t.dependencies){
      const from=nodes.get(dep),to=nodes.get(t.id);if(!from||!to)continue;const a=from.getBoundingClientRect(),b=to.getBoundingClientRect();const x1=a.right-r.left,y1=a.top+a.height/2-r.top,x2=b.left-r.left-2,y2=b.top+b.height/2-r.top;
      const line=document.createElementNS(ns,'path');line.setAttribute('d',`M${x1},${y1} C${x1+17},${y1} ${x2-17},${y2} ${x2},${y2}`);line.setAttribute('fill','none');line.setAttribute('stroke','#c2cede');line.setAttribute('stroke-width','1.2');line.setAttribute('marker-end','url(#dag-arrow)');svg.append(line);
    }
  }
  function eventTime(e){const start=S.events[0]?.timestamp;if(!number(start)||!number(e?.timestamp))return '';const sec=Math.max(0,e.timestamp-start);return `${String(Math.floor(sec/60)).padStart(2,'0')}:${String(Math.floor(sec%60)).padStart(2,'0')}`;}
  function renderEntry(entry){
    const e=entry.event||{},kind=entry.kind;const n=el('article','event '+kind);const icons={user:'user',plan:'branch',start:'robot',complete:'check',tool:'tool',error:'warning',notice:'message',handoff:'layers'};n.append(glyph(icons[kind]||'message','event-icon'));
    const content=el('div','event-content');const label=el('div','event-label');label.append(el('span','event-owner',kind==='user'?'用户任务':e.task_id?owner(e.task_id):'Runtime'),el('span','',kind==='tool'?'工具操作':entry.title),el('time','event-time',eventTime(e)));content.append(label);
    if(kind==='tool'){
      const d=el('details');d.dataset.entry=entry.key;d.open=S.open.has(entry.key);const summary=el('summary');summary.append(el('span','tool-name',entry.title));const target=entry.args.path||entry.args.query||(Array.isArray(entry.args.argv)?entry.args.argv.join(' '):'');summary.append(el('span','tool-target',target));
      let status=entry.pending?'执行中':entry.ok===false?'错误':'完成';let output=entry.output;
      if(typeof output==='string'){try{output=JSON.parse(output);}catch{/* Keep the actual tool text. */}}
      if(number(output?.returncode))status=`退出 ${output.returncode}`;
      summary.append(el('span','',status));d.append(summary);let text='参数\n'+cleanJSON(entry.args);
      if(!entry.pending)text+='\n\n'+(entry.error?'错误\n'+entry.error:'返回\n'+(typeof output==='string'?output:cleanJSON(output)));
      d.append(el('pre','',text));content.append(d);
    }else{
      content.append(el('p','event-text',entry.text||''));
      if(entry.plan){const links=el('div','plan-links');for(const t of entry.plan){const btn=el('button','',short(t.goal,34));btn.dataset.task=t.id;btn.title=t.goal;links.append(btn);}content.append(links);}
    }
    n.append(content);return n;
  }
  function renderFeed(){
    const key=[S.id,S.cursor,S.selected,S.job?.state,S.tab].join(':');if(key===S.renderKey)return;S.renderKey=key;
    const feed=$('process-panel'),scroll=feed.scrollTop;
    if(!S.job){feed.replaceChildren(...[...welcome.childNodes].map(n=>n.cloneNode(true)));hydrateIcons(feed);return;}
    const entries=S.entries.filter(e=>!S.selected||e.event?.task_id===S.selected||e.kind==='user');const frag=document.createDocumentFragment();
    if(!S.entries.some(e=>e.kind==='user'))frag.append(renderEntry({kind:'user',title:'用户任务',text:S.job.task}));
    if(entries.length>180)frag.append(el('p','muted',`显示最近 180 条可读记录；原始事件共 ${S.events.length} 条，执行详情保留完整来源。`));
    for(const entry of entries.slice(-180))frag.append(renderEntry(entry));
    const modelActive=[...S.models.values()].some(m=>m.state==='running'&&(!S.selected||m.task===S.selected));
    if(activeState(S.job.state)){
      const row=el('div','thinking-row'),dots=el('span','thinking-dots');dots.append(el('i'),el('i'),el('i'));row.append(dots,el('span','',modelActive?'模型正在响应…':'等待下一条执行记录…'));frag.append(row);
    }
    if(!entries.length&&S.selected)frag.append(el('p','feed-empty','该任务尚未产生执行记录。'));
    feed.replaceChildren(frag);if(S.follow)feed.scrollTop=feed.scrollHeight;else feed.scrollTop=scroll;
  }
  function usage(){
    if(S.job?.summary?.usage){const u=S.job.summary.usage;return {...u,prompt:u.prompt_tokens,completion:u.completion_tokens,cache:u.cache_hit_tokens??u.cached_tokens,miss:u.cache_miss_tokens,calls:u.attempts,final:true};}
    const all=[...S.models.values()],done=all.filter(x=>x.state==='completed');
    const sum=(field,alternate)=>done.length&&done.every(x=>number(x.usage?.[field]??alternate?.(x.usage)))?done.reduce((n,x)=>n+(x.usage[field]??alternate(x.usage)),0):null;
    return {prompt:sum('prompt_tokens'),completion:sum('completion_tokens'),cache:sum('prompt_cache_hit_tokens',u=>u?.prompt_tokens_details?.cached_tokens),miss:sum('prompt_cache_miss_tokens'),calls:all.length,final:false};
  }
  function seconds(){if(number(S.job?.summary?.execution_seconds))return S.job.summary.execution_seconds;const start=S.events.find(e=>e.event_type==='execution.started')?.timestamp;if(!number(start))return null;const end=S.events.findLast(e=>e.event_type==='execution.completed')?.timestamp;return Math.max(0,(number(end)?end:Date.now()/1000)-start);}
  function updateFooter(){
    const state=S.job?.state||'pending';$('status-text').replaceChildren(stateDot(state),document.createTextNode(S.job?(labels[state]||state):'准备就绪'));$('elapsed').textContent=duration(seconds());const u=usage();$('usage-footer').textContent=number(u.prompt)?`输入 ${fmt(u.prompt)} · 输出 ${fmt(u.completion)} Token`:'等待模型用量';
  }
  function renderDetails(){
    const panel=$('details-panel');if(!S.job){panel.replaceChildren(el('p','feed-empty','运行开始后显示实际配置与记录来源。'));return;}
    const c=S.job.summary?.config||S.job.config||{},u=usage(),t=S.selected?S.tasks.get(S.selected):null;
    const values=[['运行标识',S.job.run_id||'等待执行入口创建'],['工作目录',S.job.cwd],['执行框架',c.harness||'langgraph'],['执行入口',c.profile==='native'?'框架原生执行':'LangGraph → AgentOS Runtime'],['模型',c.model],['思考与容量',`${c.reasoning_effort||'high'} · 上下文与输出使用模型默认`],['命令环境',c.command_sandbox==='bubblewrap'?'bubblewrap · '+(c.command_network?'网络可用':'网络隔离'):'宿主进程'],['本次模型调用',fmt(u.calls)],['执行耗时',duration(seconds())],['产物验证',S.job.summary?.score?.status==='not_requested'?'未配置独立评分；实际自检输出见工具记录':S.job.summary?.score?.status||'执行结束后记录'],['事件记录',`${S.events.length} 条已加载`]];
    if(t)values.unshift(['当前子任务',t.goal||t.name],['实际 Agent',S.agents.get(t.id)?.id||'尚未创建'],['任务依赖',t.dependencies.join('、')||'无前置依赖']);
    const dl=el('dl','info-grid');for(const [k,v] of values){dl.append(el('dt','',k),el('dd','',v??'未记录'));}panel.replaceChildren(dl);
    if(S.job.error){const err=el('div','raw-details');err.append(el('strong','','执行信息'),el('pre','',typeof S.job.error==='string'?S.job.error:cleanJSON(S.job.error)));panel.append(err);}
    const raw=el('details','raw-details');raw.append(el('summary','','查看实际配置'),el('pre','',cleanJSON(c)));panel.append(raw);
  }
  function renderArtifacts(){
    const panel=$('artifacts-panel');$('artifact-count').hidden=!S.artifactFiles.length;$('artifact-count').textContent=S.artifactFiles.length;
    if(!S.job){panel.replaceChildren(el('p','feed-empty','完成任务后，在这里查看实际生成或修改的文件。'));return;}
    if(!S.artifactsLoaded){panel.replaceChildren(el('p','feed-empty',activeState(S.job.state)?'任务正在执行，产物将在结束后归档。':'本次没有可读取的归档产物。'));return;}
    const frag=document.createDocumentFragment();frag.append(el('p','artifact-intro',S.artifactFiles.length?`${S.artifactFiles.length} 个文件变更 · 内容来自本次运行的归档产物`:'本次没有文件变更，以下为 Agent 的实际输出。'));
    for(const f of S.artifactFiles){
      const change=(S.job.summary?.artifacts||[]).find(x=>x.path===f.path);const deleted=f.deleted||change?.after_sha256===null;
      const n=el('button','artifact-file');n.dataset.file=f.path;n.append(glyph('file'));const copy=el('div');copy.append(el('strong','',f.path),el('small','',deleted?'已删除':change?.before_sha256?'已修改':change?'已生成':'本次归档产物'));n.append(copy,glyph('external','artifact-arrow'));if(deleted||f.available===false)n.disabled=true;frag.append(n);
    }
    for(const [tid,answer] of Object.entries(S.artifactAnswers)){const card=el('section','answer-card');card.append(el('h3','',owner(tid)),el('pre','',typeof answer==='string'?answer:cleanJSON(answer)));frag.append(card);}
    panel.replaceChildren(frag);
  }
  function table(headers, rows){const t=el('table','monitor-table'),head=el('thead'),tr=el('tr');headers.forEach(h=>tr.append(el('th','',h)));head.append(tr);t.append(head);const body=el('tbody');if(!rows.length){const r=el('tr'),d=el('td','table-empty','当前运行尚无对应记录');d.colSpan=headers.length;r.append(d);body.append(r);}for(const row of rows){const r=el('tr');row.forEach(v=>r.append(el('td','',v??'—')));body.append(r);}t.append(body);return t;}
  function monitorCard(name,ico){const n=el('section','monitor-card'),h=el('h2');h.append(glyph(ico),document.createTextNode(name));n.append(h);return n;}
  function renderMonitor(){
    const root=$('monitor-content');$('monitor-subtitle').textContent=S.job?`${S.job.run_id||'准备运行'} · ${labels[S.job.state]||S.job.state}`:'选择一次真实运行，查看执行状态与用量。';
    if(!S.job){root.replaceChildren(el('div','monitor-empty','提交一个任务后，这里会展示同次运行的真实事件和统计。'));return;}
    const u=usage(),tasks=[...S.tasks.values()],frag=document.createDocumentFragment(),stats=el('div','stat-grid');
    for(const [label,value,note] of [['任务完成',`${tasks.filter(t=>t.status==='completed').length} / ${tasks.length}`,'来自 Runtime 任务状态'],['模型调用',fmt(u.calls),'包含记录到的调用尝试'],['输入 Token',fmt(u.prompt),u.final?'API 实际用量':'已返回调用的 API 用量'],['输出 Token',fmt(u.completion),'包含 API 计入的思考用量']]){const box=el('div','stat-box');box.append(el('span','stat-label',label),el('div','stat-value',value),el('div','stat-note',note));stats.append(box);}frag.append(stats);
    const agents=monitorCard('Agent 与任务执行','robot');agents.append(table(['Agent / 子任务','状态','模型调用','命令 Worker PID'],[...S.agents].map(([tid,a])=>[a.id,labels[S.tasks.get(tid)?.status]||'等待',[...S.models.values()].filter(m=>m.task===tid).length,S.commands.findLast(c=>c.task===tid)?.worker_pid??'—'])));agents.append(el('p','monitor-note','PID 来自实际命令 Worker；CPU / 内存未在此运行中采集。'));frag.append(agents);
    const refs=monitorCard('共享上下文与结果交接','layers');refs.append(table(['上下文引用','使用该引用的任务','类型'],[...S.refs.values()].map(r=>[r.name,[...r.consumers].join('、')||'—',r.name?.startsWith('result/')?'Agent 交接结果':'共享材料'])));refs.append(el('p','monitor-note',`已记录 ${S.messages} 条结果引用发送、${S.published} 次结果发布。`));frag.append(refs);
    const columns=el('div','monitor-columns'),cache=monitorCard('实际缓存用量','cpu');
    for(const [name,value] of [['命中缓存的输入 Token',u.cache],['未命中缓存的输入 Token',u.miss]]){const row=el('div','metric-pair');row.append(el('span','',name),el('strong','',fmt(value)));cache.append(row);}
    if(number(u.cache)&&number(u.prompt)&&u.prompt>0){const bar=el('div','cache-bar'),fill=el('span');fill.style.width=Math.min(100,u.cache/u.prompt*100)+'%';bar.append(fill);cache.append(bar);}
    cache.append(el('p','monitor-note','缓存计数取自模型服务返回；未提供的数据显示为 —。缓存命中的 Token 仍计入总输入。'));columns.append(cache);
    const commands=monitorCard('工具与命令记录','shield');
    const lastCommands=S.commands.slice(-6);if(!lastCommands.length)commands.append(el('p','monitor-note','当前尚无命令执行结果。'));for(const c of lastCommands){const row=el('div','activity-line');row.append(el('time','',`退出 ${c.returncode??'—'}`),el('span','',short(Array.isArray(c.argv)?c.argv.join(' '):'命令',95)));commands.append(row);}
    commands.append(el('p','monitor-note',`${S.commands.length} 次命令已返回 · ${S.job.summary?.config?.command_sandbox==='bubblewrap'||S.job.config?.command_sandbox==='bubblewrap'?'bubblewrap 命令隔离':'宿主命令执行'}`));columns.append(commands);frag.append(columns);root.replaceChildren(frag);
  }
  function updateControls(){
    const running=S.jobs.some(j=>activeState(j.state))||activeState(S.job?.state);
    $('submit-task').disabled=S.submitting||running||!S.config||!S.config.credential_configured;
    $('submit-task').querySelector('span').textContent=S.submitting?'正在提交':running?'任务运行中':'运行任务';
    $('cancel-task').hidden=!S.job||!activeState(S.job.state);$('cancel-task').disabled=S.job?.state==='cancelling';
    $('execution-note').textContent=running?'执行中可切换 Agent、查看工具记录 · 配置仅作用于新任务':'配置作用于新任务 · Ctrl / ⌘ + Enter 运行';
  }
  function renderAll(){
    renderSidebar();renderDAG();renderFeed();renderDetails();renderArtifacts();if(S.view==='monitor')renderMonitor();
    const t=S.selected?S.tasks.get(S.selected):null;$('feed-title').textContent=t?(t.id==='planner'?'规划 Agent':t.name):S.job?'全部 Agent 的执行记录':'开始一次协作';$('feed-title').title=t?.goal||'';
    $('feed-subtitle').textContent=t?`${S.agents.get(t.id)?.id||'等待调度'} · ${t.dependencies.length?'依赖 '+t.dependencies.join('、'):'无前置依赖'}`:S.job?(S.job.run_id||'正在准备执行环境'):'提交任务，查看 Agent 的实际执行过程与产物。';
    const state=t?.status||S.job?.state;$('run-state').className='state-badge '+(state||'');$('run-state').textContent=labels[state]||'等待任务';updateControls();updateFooter();
  }
  function setTab(tab){S.tab=tab;document.querySelectorAll('[data-tab]').forEach(n=>{const on=n.dataset.tab===tab;n.classList.toggle('active',on);n.setAttribute('aria-selected',String(on));});for(const id of ['process','details','artifacts'])$(id+'-panel').hidden=id!==tab;$('follow-feed').hidden=tab!=='process';}
  function setView(view){S.view=view;document.querySelectorAll('[data-view]').forEach(n=>{const on=n.dataset.view===view;n.classList.toggle('active',on);n.setAttribute('aria-pressed',String(on));});$('workbench-view').hidden=view!=='workbench';$('monitor-view').hidden=view!=='monitor';if(view==='monitor')renderMonitor();else requestAnimationFrame(drawWires);}
  async function openFile(path){
    if(!S.id)return;try {const data=await api(`/jobs/${encodeURIComponent(S.id)}/artifacts/file?path=${encodeURIComponent(path)}`);$('file-title').textContent=data.path||path;$('file-content').textContent=data.content??'无法预览此文件。';$('file-meta').textContent=[data.sha256?'SHA-256 '+short(data.sha256,18):'本次运行归档文件',data.truncated?'预览已截断':''].filter(Boolean).join(' · ');$('file-dialog').showModal();}catch(e){toast(e.message);}
  }
  $('task-form').addEventListener('submit',async event=>{
    event.preventDefault();if($('submit-task').disabled)return;const task=$('task-input').value.trim(),cwd=$('cwd-input').value.trim();if(!task||!cwd)return;
    S.submitting=true;updateControls();try {
      const job=await api('/jobs',{method:'POST',body:JSON.stringify({task,cwd,config:{model:$('model-input').value.trim(),base_url:$('base-url-input').value.trim(),harness:$('harness-input').value}})});
      S.jobs.unshift(job);await selectJob(job);await refreshJobs();
    }catch(e){toast(e.message);}finally{S.submitting=false;updateControls();}
  });
  $('task-input').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter'){event.preventDefault();$('task-form').requestSubmit();}});
  $('new-task').addEventListener('click',()=>selectJob(null));
  $('refresh-jobs').addEventListener('click',()=>refreshJobs().catch(e=>toast(e.message)));
  $('cancel-task').addEventListener('click',async()=>{if(!S.id)return;try{S.job=await api(`/jobs/${encodeURIComponent(S.id)}/cancel`,{method:'POST',body:'{}'});renderAll();}catch(e){toast(e.message);}});
  document.addEventListener('click',event=>{
    const task=event.target.closest('[data-task]');if(task){setSelected(task.dataset.task);return;}
    const job=event.target.closest('[data-job]');if(job){const item=S.jobs.find(j=>j.id===job.dataset.job);if(item)selectJob(item).catch(e=>toast(e.message));return;}
    const tab=event.target.closest('[data-tab]');if(tab){setTab(tab.dataset.tab);return;}
    const view=event.target.closest('[data-view]');if(view){setView(view.dataset.view);return;}
    const prompt=event.target.closest('[data-prompt]');if(prompt){$('task-input').value=prompt.dataset.prompt.replaceAll('\\n','\n');$('task-input').focus();return;}
    const file=event.target.closest('[data-file]');if(file)openFile(file.dataset.file);
  });
  $('process-panel').addEventListener('toggle',event=>{const n=event.target;if(n.matches('details[data-entry]')&&n.isConnected){if(n.open)S.open.add(n.dataset.entry);else S.open.delete(n.dataset.entry);}},true);
  $('process-panel').addEventListener('scroll',()=>{const n=$('process-panel');S.follow=n.scrollHeight-n.scrollTop-n.clientHeight<65;},{passive:true});
  $('follow-feed').addEventListener('click',()=>{S.follow=true;$('process-panel').scrollTop=$('process-panel').scrollHeight;});
  $('settings-button').addEventListener('click',()=>{$('settings-dialog').showModal();});
  $('close-settings').addEventListener('click',()=>$('settings-dialog').close());
  $('settings-form').addEventListener('submit',event=>{event.preventDefault();if(!$('settings-form').reportValidity())return;$('settings-dialog').close();toast('模型服务配置已保存，将用于下一次任务。',true);});
  $('close-file').addEventListener('click',()=>$('file-dialog').close());
  $('back-to-dag').addEventListener('click',()=>setView('workbench'));
  window.addEventListener('resize',()=>requestAnimationFrame(drawWires));
  async function boot(){
    hydrateIcons();renderAll();
    try{
      S.config=await api('/config');const c=S.config.defaults||{};$('cwd-input').value=c.cwd||'';$('model-input').value=c.model||'deepseek-flash';$('base-url-input').value=c.base_url||'https://api.deepseek.com';
      const frameworks=S.config.harnesses||[{id:'langgraph',name:'LangGraph'}];$('harness-input').replaceChildren(...frameworks.map(h=>{const o=el('option','',h.name||h.id);o.value=h.id;return o;}));$('harness-input').value=c.harness||'langgraph';
      $('credential-note').textContent=S.config.credential_configured?'模型凭据已配置':'模型凭据未配置';$('settings-credential').textContent=S.config.credential_configured?'已在本地服务配置':'未配置';
      if(!S.config.credential_configured)toast('请在本地服务配置模型凭据后启动任务，页面不会读取或显示密钥。');
      await refreshJobs();const selected=new URLSearchParams(location.hash.slice(1)).get('job');const job=S.jobs.find(x=>x.id===selected)||S.jobs[0];if(job)await selectJob(job);else renderAll();
    }catch(e){connection(false);toast('连接本地服务失败：'+e.message);}
    tick();setInterval(updateFooter,1000);
  }
  boot();
})();
