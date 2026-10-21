"""Post-pilot stability fixes. Synthetic model responses, no API or SWE reruns."""
import json
import sys
from pathlib import Path

import pytest

from agentos.execution.backend import _usage_metrics
from agentos.execution.search import MAX_OUTPUT_BYTES, search_files
from agentos.execution.service import execute
from agentos.llm.base import LLMResponse


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME',str(tmp_path/'config'))
    root=tmp_path/'workspace';root.mkdir()
    (root/'source.py').write_text('def locate_symbol():\n    return "evidence"\n')
    return root


def config(tmp_path, **overrides):
    return {'model':'synthetic','base_url':'http://unused.test','output':str(tmp_path/'runs'),
            'profile':'runtime','thinking':'enabled','reasoning_effort':'high',
            'max_tokens':None,'max_steps':24,'max_tasks':6,'max_parallel':2,'retries':1,
            'command_sandbox':'none',**overrides}


def plan(read_only=False, goal='Complete only this assigned subtask'):
    return {'role':'assistant','content':json.dumps({'tasks':[{'id':'work','goal':goal,'read_only':read_only}]}),
            'reasoning_content':'complete planner thinking'}


def invocation(cid,name,args):
    return {'id':cid,'type':'function','function':{'name':name,'arguments':args if isinstance(args,str) else json.dumps(args)}}


def tools(*calls):
    return {'role':'assistant','content':None,'reasoning_content':'complete worker thinking with native calls',
            'tool_calls':list(calls)}


def final():
    return {'role':'assistant','content':'Assigned goal complete; handing off evidence.', 'reasoning_content':'complete final thought'}


class Backend:
    def __init__(self,planner,worker):
        self.planner=planner;self.worker=worker;self.requests=[];self.counts={}
    async def generate(self,request):
        self.requests.append(request)
        index=self.counts.get(request.task_id,0);self.counts[request.task_id]=index+1
        values=self.planner if request.task_id=='planner' else self.worker
        message=values(index,request) if callable(values) else values[index]
        raw={'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}
        finish='tool_calls' if message.get('tool_calls') else 'stop'
        metrics,total=_usage_metrics(raw,model='synthetic',elapsed=0,finish_reason=finish)
        return LLMResponse(message.get('content') or '',total,'synthetic',0,metrics=metrics,
            metadata={'assistant_message':message,'usage':raw,'finish_reason':finish})


def events(result):
    return [json.loads(line) for line in (Path(result['output'])/'events.jsonl').read_text().splitlines()]


@pytest.mark.parametrize('invalid', [
    '<DSML calls><invoke name="run_command">touch SHOULD_NOT_EXIST</invoke></DSML>',
    '{"tasks":[]}',
    '{"tasks":[{"id":"work","goal":"g","dependencies":["missing"]}]}',
])
async def test_planner_one_correction_preserves_history_usage_and_no_side_effects(workspace,tmp_path,invalid):
    original={'role':'assistant','content':invalid,'reasoning_content':'original malformed-plan thinking'}
    backend=Backend([original,plan()], [final()])
    result=await execute('Overall issue',cwd=workspace,overrides=config(tmp_path),backend=backend)
    assert result['workflow_status']=='completed',result['error']
    assert backend.counts=={'planner':2,'work':1}
    assert original in backend.requests[1].messages
    assert all(not request.tools for request in backend.requests if request.task_id=='planner')
    assert 'planning phase has no tools' in backend.requests[0].messages[-1]['content']
    assert result['protocol']['planner_format_corrections']==1
    assert result['usage']['retries']==0 and result['usage']['prompt_tokens']==30
    stream=events(result)
    assert len([e for e in stream if e['event_type']=='harness.protocol_correction'])==1
    assert not any(e['event_type']=='execution.tool_started' for e in stream)
    assert not (workspace/'SHOULD_NOT_EXIST').exists()


async def test_planner_format_retry_is_bounded(workspace,tmp_path):
    bad={'role':'assistant','content':'not a plan','reasoning_content':'full thought'}
    backend=Backend([bad,bad], [])
    result=await execute('task',cwd=workspace,overrides=config(tmp_path),backend=backend)
    assert result['workflow_status']=='failed'
    assert backend.counts=={'planner':2}
    assert result['protocol']['planner_format_corrections']==1
    assert result['usage']['prompt_tokens']==20
    assert not (Path(result['output'])/'plan.json').exists()


async def test_mixed_tool_batch_executes_valid_side_effect_once_and_replays_full_history(workspace,tmp_path):
    good=invocation('good','run_command',{'argv':[sys.executable,'-c',"from pathlib import Path; p=Path('effect.txt'); p.write_text((p.read_text() if p.exists() else '')+'x')"]})
    bad=invocation('bad','run_command','{"argv":["python", "unterminated\'')
    original=tools(good,bad)
    def worker(index,request):
        if index==0:return original
        if index==1:
            assert original in request.messages
            observations=[m for m in request.messages if m['role']=='tool']
            assert [m['tool_call_id'] for m in observations]==['good','bad']
            assert json.loads(observations[0]['content'])['ok']
            error=json.loads(observations[1]['content'])
            assert not error['ok'] and error['executed'] is False
            return tools(invocation('fixed','run_command',{'argv':[sys.executable,'-c','print("corrected")']}))
        return final()
    backend=Backend([plan()],worker)
    result=await execute('task',cwd=workspace,overrides=config(tmp_path),backend=backend)
    assert result['workflow_status']=='completed',result['error']
    assert (workspace/'effect.txt').read_text()=='x'
    stream=events(result)
    assert len([e for e in stream if e['event_type']=='execution.command_result'])==2
    assert result['protocol']['worker_argument_errors']==1
    assert result['usage']['retries']==0 and result['usage']['attempts']==4
    assert result['config']['max_steps']==24 and result['config']['max_tokens'] is None
    assert result['config']['thinking']=='enabled' and result['config']['reasoning_effort']=='high'


@pytest.mark.parametrize('name,args', [('run_command',{'argv':'not an array'}),('read_file',{'offset':100,'path':'source.py'}),('read_file',[]),('search_files',{'max_results':10})])
async def test_schema_errors_are_matching_observations_without_tool_execution(workspace,tmp_path,name,args):
    original=tools(invocation('invalid',name,args if isinstance(args,dict) else json.dumps(args)))
    def worker(index,request):
        if index==0:return original
        assert original in request.messages
        last=request.messages[-1]
        assert last['role']=='tool' and last['tool_call_id']=='invalid'
        assert json.loads(last['content'])['executed'] is False
        return final()
    result=await execute('task',cwd=workspace,overrides=config(tmp_path),backend=Backend([plan()],worker))
    assert result['workflow_status']=='completed',result['error']
    assert not any(e['event_type']=='execution.tool_started' for e in events(result))


@pytest.mark.parametrize('mode',['missing','duplicate'])
async def test_unpairable_ids_fail_before_any_side_effect(workspace,tmp_path,mode):
    good=invocation('same','write_file',{'path':'bad.txt','content':'should not execute'})
    bad=invocation('same' if mode=='duplicate' else '', 'read_file',{'path':'source.py'})
    result=await execute('task',cwd=workspace,overrides=config(tmp_path),backend=Backend([plan()],[tools(good,bad)]))
    assert result['workflow_status']=='failed'
    assert not (workspace/'bad.txt').exists()
    assert not any(e['event_type']=='execution.tool_started' for e in events(result))


async def test_errors_consume_original_24_steps_and_quality_stays_independent(workspace,tmp_path):
    def worker(index,request):
        if index==0:return tools(invocation('write','write_file',{'path':'answer.txt','content':'correct'}))
        return tools(invocation('bad-'+str(index),'run_command','{"argv":'))
    async def scorer(cwd,output):
        return {'status':'completed','passed':(cwd/'answer.txt').read_text()=='correct'}
    backend=Backend([plan()],worker)
    result=await execute('task',cwd=workspace,overrides=config(tmp_path),backend=backend,scorer=scorer)
    assert result['workflow_status']=='failed' and result['score']['passed'] is True
    assert backend.counts['work']==24 and result['usage']['attempts']==25
    assert result['protocol']['worker_argument_errors']==23
    assert (Path(result['output'])/'artifacts/files/answer.txt').read_text()=='correct'


def test_search_literal_paths_permissions_and_no_writes(workspace,tmp_path):
    (workspace/'literal.txt').write_text('a123b\na.*b\n')
    outside=tmp_path/'outside.txt';outside.write_text('needle outside secret')
    (workspace/'outside-link').symlink_to(outside)
    hidden=workspace/'.git';hidden.mkdir();(hidden/'hidden.txt').write_text('needle hidden')
    before=(workspace/'literal.txt').read_bytes()
    result=search_files(workspace,'a.*b')
    assert len(result['matches'])==1 and result['matches'][0]['line']==2
    assert (workspace/'literal.txt').read_bytes()==before
    assert not search_files(workspace,'needle')['matches']
    for path in ('../outside.txt','outside-link'):
        with pytest.raises(ValueError,match='escapes'):search_files(workspace,'needle',path=path)


def test_search_bounds_and_skips_are_explicit(workspace):
    (workspace/'many.txt').write_text(('needle '+('界'*500)+'\n')*150)
    (workspace/'binary.dat').write_bytes(b'needle\x00binary')
    (workspace/'large.txt').write_bytes(b'x'*(1024*1024+1))
    result=search_files(workspace,'needle',max_results=100)
    assert result['truncated'] and 'output_byte_limit' in result['truncation_reasons']
    assert len(json.dumps(result,ensure_ascii=False).encode())<=MAX_OUTPUT_BYTES
    assert result['skipped']['binary_or_non_utf8']==1 and result['skipped']['large_files']==1
    assert any(m['excerpt_truncated'] for m in result['matches'])
    assert search_files(workspace,'needle',max_results=1)['truncation_reasons']==['result_limit_reached']
    with pytest.raises(ValueError):search_files(workspace,'needle',max_results=100000)


@pytest.mark.parametrize('context_mode',['off','on'])
async def test_readonly_search_and_assigned_goal_boundary_in_both_arms(workspace,tmp_path,context_mode):
    original=tools(invocation('search','search_files',{'query':'def locate_symbol','file_glob':'*.py'}))
    def worker(index,request):
        offered={t['function']['name'] for t in request.tools}
        assert offered=={'list_files','read_file','search_files'}
        instruction=next(m['content'] for m in request.messages if m['role']=='user')
        assert 'Overall issue (background only)' in instruction
        assert 'Assigned subtask (your current work scope)' in instruction
        assert 'Locate the symbol and hand off findings only' in instruction
        assert 'leave downstream fixes and reviews' in instruction
        if index==0:return original
        observation=json.loads(request.messages[-1]['content'])
        assert observation['ok'] and observation['output']['matches'][0]['path']=='source.py'
        return final()
    result=await execute('Fix the whole issue eventually',cwd=workspace,
        overrides=config(tmp_path,context_optimization=context_mode),
        backend=Backend([plan(True,'Locate the symbol and hand off findings only')],worker))
    assert result['workflow_status']=='completed',result['error']
    assert result['artifacts']==[]
    tool_events=[e for e in events(result) if e['event_type']=='execution.tool_finished']
    assert len(tool_events)==1 and tool_events[0]['data']['tool']=='search_files'


def test_search_skips_special_files_and_keeps_unicode_match_excerpt(workspace):
    import os
    os.mkfifo(workspace/'fifo')
    (workspace/'unicode.txt').write_text(('ß'*400)+' TARGET '+('界'*400)+'\n')
    result=search_files(workspace,'target',case_sensitive=False)
    assert result['skipped']['special_files']==1
    assert len(result['matches'])==1 and 'TARGET' in result['matches'][0]['text']
    assert result['matches'][0]['excerpt_truncated']
