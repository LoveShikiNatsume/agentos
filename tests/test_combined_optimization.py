"""New combined-arm contract tests, with explicit synthetic model responses only."""
import asyncio
import csv
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentos.context.manager import ContextManager
from agentos.context.types import ContextAccessDenied
from agentos.execution.backend import _usage_metrics
from agentos.execution.bench import report, run_suite
from agentos.execution.config import resolve_config
from agentos.execution.context_policy import ExecutionContextManager
from agentos.execution.service import execute
from agentos.llm.base import LLMResponse


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path/'config'))
    root=tmp_path/'workspace';root.mkdir()
    (root/'a.txt').write_text('A repository finding\n')
    (root/'b.txt').write_text('B independent repository finding\n')
    return root


def options(tmp_path, mode='legacy'):
    return {'model':'synthetic','base_url':'http://unused.test','output':str(tmp_path/'runs'),
            'profile':'runtime','context_optimization':mode,'max_parallel':2,
            'thinking':'enabled','reasoning_effort':'high','max_tokens':None,
            'retries':1,'command_sandbox':'none'}


def seed(context):
    context.add_shared('policy','Permanent instructions')
    context.add_shared('task','Unmodified user request')
    context.add_shared('result/needed','Actual dependency result')
    context.add_shared('result/other','Other finished branch result α')
    context.add_private('agent-a','own private material')
    context.add_private('agent-b','foreign private material')
    denied=context.add_shared_ref('denied','unreadable shared material')
    context._refs['denied']=replace(denied,permissions=frozenset())


@pytest.mark.parametrize('mode', ['off','on','legacy'])
def test_selection_permissions_and_complete_native_messages(mode):
    context=ExecutionContextManager(mode);seed(context)
    history=[{'role':'user','content':'original instruction'},
             {'role':'assistant','content':None,'reasoning_content':'full provider thought',
              'tool_calls':[{'id':'call1','type':'function','function':{'name':'read_file','arguments':'{"path":"a.txt"}'}}]},
             {'role':'tool','tool_call_id':'call1','content':'complete tool output'}]
    prepared=context.prepare_context('agent-a',history,context_ref_names=['policy','task','result/needed'])
    assert prepared.messages[1:]==history
    prefix=prepared.messages[0]['content']
    assert 'Permanent instructions' in prefix and 'Unmodified user request' in prefix
    assert 'own private material' in prefix
    assert 'foreign private material' not in prefix and 'unreadable shared material' not in prefix
    audit=prepared.selection_details
    if mode=='off':
        assert 'result/other' in prepared.shared_refs
        assert audit['selection']=='all_shared' and audit['filtered_ref_count']==0
    else:
        assert 'result/other' not in prepared.shared_refs
        assert audit['filtered_refs']==['result/other']
        assert audit['filtered_shared_bytes']==len('Other finished branch result α'.encode())
    assert audit['lossy_compression'] is False and audit['segment_budget'] is None
    assert 'not model tokens' in audit['byte_metric_scope']
    if mode!='off':
        with pytest.raises(ContextAccessDenied):
            context.prepare_context('agent-a',history,context_ref_names=['denied'])


def test_legacy_preserves_previous_request_organization():
    previous=ContextManager();current=ExecutionContextManager('legacy')
    for manager in (previous,current):seed(manager)
    kwargs={'context_ref_names':['policy','task','result/needed']}
    old=previous.prepare_context('agent-a',[{'role':'user','content':'task'}],**kwargs)
    new=current.prepare_context('agent-a',[{'role':'user','content':'task'}],**kwargs)
    assert (old.messages,old.prefix_id,old.shared_refs)==(new.messages,new.prefix_id,new.shared_refs)


def test_stable_prefix_organization_without_random_baseline():
    def build(mode,names):
        manager=ExecutionContextManager(mode)
        manager.add_shared('policy','instructions');manager.add_shared('task','task')
        for name in names:manager.add_shared(name,name+' actual result')
        return manager
    first=build('on',['result/z','result/a']);second=build('on',['result/a','result/z'])
    refs=['policy','task','result/a','result/z']
    assert first.prepare_context('a',[],context_ref_names=refs).messages==second.prepare_context('a',[],context_ref_names=refs).messages
    baseline=build('off',['result/z','result/a'])
    assert baseline.prepare_context('a',[]).messages==baseline.prepare_context('a',[]).messages


class BranchBackend:
    def __init__(self,usage=True):self.requests=[];self.counts={};self.with_usage=usage
    async def generate(self, request):
        self.requests.append(request)
        task=request.task_id;turn=self.counts[task]=self.counts.get(task,0)+1
        if task=='planner':
            content=json.dumps({'tasks':[
                {'id':'a','goal':'Read a.txt','read_only':True},
                {'id':'b','goal':'Read b.txt','read_only':True},
                {'id':'consumer','goal':'Use only branch a finding','dependencies':['a'],'read_only':True}]})
            assistant={'role':'assistant','content':content,'reasoning_content':'planner thought'};finish='stop'
        elif turn==1:
            if task=='a':await asyncio.sleep(.035)  # b actually publishes first
            assistant={'role':'assistant','content':None,'reasoning_content':'complete thought '+task,
                'tool_calls':[{'id':'read-'+task,'type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'b.txt' if task=='b' else 'a.txt'})}}]}
            content='';finish='tool_calls'
        else:
            assistants=[m for m in request.messages if m['role']=='assistant']
            assert assistants[-1]['reasoning_content']=='complete thought '+task
            results=[m for m in request.messages if m['role']=='tool']
            assert results[-1]['tool_call_id']=='read-'+task
            content=task+': '+json.loads(results[-1]['content'])['output']
            assistant={'role':'assistant','content':content,'reasoning_content':'final thought'};finish='stop'
        raw={'prompt_tokens':128,'completion_tokens':16,'total_tokens':144,
             'prompt_cache_hit_tokens':64,'prompt_cache_miss_tokens':64} if self.with_usage else None
        metrics,tokens=_usage_metrics(raw,model='synthetic',elapsed=0,finish_reason=finish)
        return LLMResponse(content,tokens,'synthetic',0,metrics=metrics,
                           metadata={'usage':raw,'assistant_message':assistant,'finish_reason':finish})


def event_rows(result):
    return [json.loads(line) for line in (Path(result['output'])/'events.jsonl').read_text().splitlines()]


async def test_real_runtime_requests_and_cache_feedback_differ_only_by_policy(workspace,tmp_path):
    results={};backends={}
    for mode in ('off','on'):
        backend=BranchBackend();backends[mode]=backend
        results[mode]=await execute('Read the repository findings',cwd=workspace,overrides=options(tmp_path,mode),backend=backend)
        assert results[mode]['workflow_status']=='completed',results[mode]['error']
    off=next(r for r in backends['off'].requests if r.task_id=='consumer')
    on=next(r for r in backends['on'].requests if r.task_id=='consumer')
    assert 'result/b' in off.context_refs and 'result/b' not in on.context_refs
    assert 'B independent repository finding' in off.messages[0]['content']
    assert 'B independent repository finding' not in on.messages[0]['content']
    for request in (off,on):
        assert {'policy','task','result/a'} <= set(request.context_refs)
        assert request.metadata['context_selection']['selected_refs']==request.context_refs
    assert on.metadata['context_selection']['filtered_shared_bytes']>0
    for mode,result in results.items():
        feedback=[e['data'] for e in event_rows(result) if e['event_type']=='execution.cache_feedback']
        assert len(feedback)==len(backends[mode].requests)
        assert all(e['cache_metrics_available'] for e in feedback)
        assert all(e['feedback_applied']==(mode=='on') for e in feedback)
        assert all(e['cache_hit_tokens']==64 for e in feedback)
        decisions=[e['data'] for e in event_rows(result) if e['event_type']=='scheduler.decision']
        assert decisions and all(d['strategy']==('cache_affinity' if mode=='on' else 'critical_path') for d in decisions)
        policies=result['execution_policies'];assert policies['max_parallel']==2 and policies['failure_policy']=='isolate'
        assert policies['scheduler_policy']=='existing_runtime_default' and policies['api_retries']==1
        rows=[json.loads(x) for x in (Path(result['output'])/'calls.jsonl').read_text().splitlines()]
        assert all(row['prefix_id'] and row['context_selection'] for row in rows)
        assert result['usage']['prompt_tokens']==128*len(backends[mode].requests)  # API fixture values, never byte estimates
        assert result['usage']['cache_hit_tokens']==result['usage']['cache_miss_tokens']==64*len(backends[mode].requests)
        assert result['usage']['cache_miss_source']=='provider_usage'
    unchanged=lambda r:{k:v for k,v in r['execution_policies'].items() if k!='cache_affinity_feedback_enabled'}
    assert unchanged(results['off'])==unchanged(results['on'])


async def test_unknown_cache_usage_never_creates_feedback(workspace,tmp_path):
    result=await execute('read files',cwd=workspace,overrides=options(tmp_path,'on'),backend=BranchBackend(usage=False))
    assert result['workflow_status']=='completed'
    assert result['usage']['prompt_tokens'] is None and not result['usage']['usage_complete']
    feedback=[e['data'] for e in event_rows(result) if e['event_type']=='execution.cache_feedback']
    assert all(not e['feedback_applied'] and e['prefix_samples']==0 and e['cache_hit_tokens'] is None for e in feedback)


async def test_prompt_describes_selected_sandbox_equally(workspace,tmp_path):
    backend=BranchBackend()
    result=await execute('read only',cwd=workspace,overrides={**options(tmp_path,'on'),'command_sandbox':'bubblewrap'},backend=backend)
    assert result['workflow_status']=='completed'
    text=next(r for r in backend.requests if r.task_id=='a').messages[-1]['content']
    assert 'configured bubblewrap OS sandbox' in text and 'network access is disabled' in text
    assert 'not a sandbox' not in text


async def test_combined_arms_frozen_config_order_and_csv(workspace,tmp_path,monkeypatch):
    from agentos.execution.service import execute as real_execute
    manifest=tmp_path/'suite.json'
    settings={k:v for k,v in options(tmp_path).items() if k not in ('output','context_optimization')}
    manifest.write_text(json.dumps({'schema':'agentos.bench.v1','config':settings,'tasks':[
        {'id':'first','cwd':'workspace','task':'read findings'},
        {'id':'second','cwd':'workspace','task':'read findings again'}]}))
    observed=[]
    async def tracking(task,**kwargs):
        observed.append((kwargs['provenance']['arm'],dict(kwargs['overrides'])))
        # Mutating source project defaults after first run must not change the second arm.
        if len(observed)==1:(workspace/'.agentos.toml').write_text('max_parallel=9\nretries=9\n')
        return await real_execute(task,**kwargs)
    monkeypatch.setattr('agentos.execution.bench.execute',tracking)
    batch=await run_suite(str(manifest),compare='baseline,optimized',output=str(tmp_path/'batches'),backend_factory=BranchBackend)
    assert [row['arm'] for row in batch['runs']]==['baseline','optimized','optimized','baseline']
    assert batch['profiles']==['runtime']
    assert all(row['profile']=='runtime' for row in batch['runs'])
    assert [row['context_optimization'] for row in batch['runs']]==['off','on','on','off']
    for _,config in observed:assert config['max_parallel']==2 and config['retries']==1 and config['reasoning_effort']=='high'
    normalize=lambda c:{k:v for k,v in c.items() if k not in ('context_optimization','output')}
    assert all(normalize(config)==normalize(observed[0][1]) for _,config in observed)
    rows=list(csv.DictReader(io.StringIO(report(batch['output']))))
    assert [r['arm'] for r in rows]==['baseline','optimized','optimized','baseline']
    assert all(r['profile']=='runtime' for r in rows)
    assert all(int(r['prompt_tokens'])==int(r['cache_hit_tokens'])+int(r['cache_miss_tokens']) for r in rows)
    order=[json.loads(x) for x in (Path(batch['output'])/'execution-order.jsonl').read_text().splitlines()]
    assert [r['arm'] for r in order if r['state']=='started']==[r['arm'] for r in rows]
    first_runs=batch['runs'][:2]
    starts=[json.loads((Path(r['output'])/'start.json').read_text())['files'] for r in first_runs]
    assert starts[0]==starts[1]
    for entry in batch['runs']:
        assert json.loads((Path(entry['output'])/'start.json').read_text())['files']==starts[0]


async def test_old_native_runtime_comparison_keeps_legacy(workspace,tmp_path):
    manifest=tmp_path/'suite.json'
    manifest.write_text(json.dumps({'schema':'agentos.bench.v1','config':{k:v for k,v in options(tmp_path).items() if k!='output'},
        'tasks':[{'id':'one','cwd':'workspace','task':'read findings'}]}))
    batch=await run_suite(str(manifest),compare='native,runtime',output=str(tmp_path/'batches'),backend_factory=BranchBackend)
    assert [r['profile'] for r in batch['runs']]==['native','runtime']
    assert all(r['context_optimization']=='legacy' for r in batch['runs'])
    assert resolve_config(workspace,{'model':'test','base_url':'http://unused.test'})['context_optimization']=='legacy'


async def test_known_tokens_but_unknown_cache_are_not_cache_misses(workspace,tmp_path):
    class NoCacheBackend(BranchBackend):
        async def generate(self,request):
            response=await super().generate(request)
            raw={k:v for k,v in response.metadata['usage'].items() if 'cache' not in k}
            response.metadata['usage']=raw
            response.metrics,_=_usage_metrics(raw,model='synthetic',elapsed=0,finish_reason=response.metadata['finish_reason'])
            return response
    result=await execute('read files',cwd=workspace,overrides=options(tmp_path,'on'),backend=NoCacheBackend())
    assert result['workflow_status']=='completed' and result['usage']['usage_complete']
    assert result['usage']['prompt_tokens']>0 and result['usage']['cached_tokens'] is None
    assert result['usage']['cache_miss_tokens'] is None
    feedback=[e['data'] for e in event_rows(result) if e['event_type']=='execution.cache_feedback']
    assert all(not e['feedback_applied'] and not e['cache_metrics_available'] for e in feedback)


def test_cli_switch_and_unsupported_native_combination(workspace):
    from agentos.cli import build_parser
    args=build_parser().parse_args(['exec','task','--context-optimization','on'])
    assert args.context_optimization=='on'
    with pytest.raises(ValueError,match='profile=runtime'):
        resolve_config(workspace,{'model':'m','base_url':'http://unused.test','profile':'native','context_optimization':'on'})


async def test_concurrent_unknown_call_does_not_borrow_other_calls_feedback(tmp_path):
    from agentos.execution.context_policy import ExecutionModelGateway
    from agentos.execution.records import Records
    from agentos.llm.base import LLMRequest
    class MixedBackend:
        async def generate(self,request):
            known=request.agent_id=='known'
            await asyncio.sleep(.01 if known else .025)
            raw={'prompt_tokens':20,'completion_tokens':5,'prompt_cache_hit_tokens':10,'prompt_cache_miss_tokens':10} if known else None
            metrics,total=_usage_metrics(raw,model='synthetic',elapsed=0,finish_reason='stop')
            return LLMResponse('done',total,'synthetic',0,metrics=metrics,metadata={'usage':raw})
    records=Records(tmp_path,'mixed')
    gateway=ExecutionModelGateway(MixedBackend(),records,'on')
    await asyncio.gather(*(gateway.generate(LLMRequest(name,[{'role':'user','content':'task'}],task_id=name,prefix_id='same')) for name in ('known','unknown')))
    data={event.task_id:event.data for event in records.read(event_type='execution.cache_feedback')}
    assert data['known']['feedback_applied']
    assert not data['unknown']['feedback_applied'] and data['unknown']['prefix_samples']==1
    assert data['unknown']['cache_hit_tokens'] is None
