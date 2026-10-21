"""Only local HTTP and explicit execution stubs; no model, benchmark or Docker."""
import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agentos.events.types import RuntimeEvent
from agentos.workbench.server import STORE, PREVIEW_BYTES, create_app


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME',str(tmp_path/'config'))
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    root=tmp_path/'workspace';root.mkdir()
    return root


def same_origin(client):
    return {'Origin':str(client.make_url('/')).rstrip('/')}


async def completed_job(client,job_id):
    for _ in range(100):
        data=await (await client.get('/api/workbench/jobs/'+job_id)).json()
        if data['state'] not in {'queued','running'}:return data
        await asyncio.sleep(.005)
    raise AssertionError('job did not finish')


class ExecutionStub:
    def __init__(self,gate=None):self.calls=[];self.gate=gate;self.cancelled=False
    async def __call__(self,task,*,cwd,overrides,listener):
        self.calls.append((task,cwd,overrides))
        run_id='run-actual-'+str(len(self.calls))
        output=Path(overrides['output'])/run_id;output.mkdir()
        log=output/'events.jsonl'
        for index in range(3):
            event=RuntimeEvent(event_type='task.started' if index==0 else 'tool.call_finished',
                               task_id='work',data={'run_id':run_id,'index':index}).to_dict()
            with log.open('a') as stream:stream.write(json.dumps(event)+'\n')
            listener(event)
        if self.gate:
            try:await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled=True
                raise
        content='actual archived output\n'
        target=output/'artifacts/files/report.txt';target.parent.mkdir(parents=True);target.write_text(content)
        sha=hashlib.sha256(content.encode()).hexdigest()
        changes=[{'path':'report.txt','before_sha256':None,'after_sha256':sha}]
        (output/'artifacts.json').write_text(json.dumps({'changes':changes,'answers':{'work':'actual handoff'}}))
        summary={'run_id':run_id,'workflow_status':'completed','error':None,'answers':{'work':'actual handoff'},
                 'usage':{'prompt_tokens':None,'completion_tokens':None,'usage_complete':False},
                 'output':str(output),'artifacts':changes,'config':overrides,'execution_seconds':.1}
        (output/'summary.json').write_text(json.dumps(summary))
        return summary


async def test_config_credentials_and_missing_key_are_safe(workspace,tmp_path,monkeypatch):
    async with TestClient(TestServer(create_app(workspace,executor=ExecutionStub()))) as client:
        config=await (await client.get('/api/workbench/config')).json()
        assert config['defaults']['model']=='deepseek-flash'
        assert config['defaults']['thinking']=='enabled' and config['defaults']['reasoning_effort']=='high'
        assert config['defaults']['max_tokens'] is None and config['defaults']['max_parallel']==2
        assert config['harnesses']==[{'id':'langgraph','name':'LangGraph'}]
        assert config['credential_configured'] is False and 'api_key_env' not in config['defaults']
        response=await client.post('/api/workbench/jobs',json={'task':'task'},headers=same_origin(client))
        assert response.status==400 and 'credential' in (await response.json())['error']
    env=tmp_path/'private.env';env.write_text('DEEPSEEK_API_KEY=server-secret-value\nOTHER_PRIVATE_TOKEN=other-secret\n')
    async with TestClient(TestServer(create_app(workspace,env_file=env,executor=ExecutionStub()))) as client:
        response=await client.get('/api/workbench/config');body=await response.text()
        assert 'server-secret-value' not in body and 'other-secret' not in body
        assert json.loads(body)['credential_configured'] is True
        assert os.environ.get('OTHER_PRIVATE_TOKEN') is None


async def test_jobs_real_id_events_artifacts_history_and_monitor(workspace,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','server-secret-value')
    executor=ExecutionStub();app=create_app(workspace,executor=executor)
    async with TestClient(TestServer(app)) as client:
        response=await client.post('/api/workbench/jobs',json={'task':'write report','cwd':'.','config':{'model':'chosen','base_url':'https://api.deepseek.com','harness':'langgraph'}},headers=same_origin(client))
        assert response.status==202
        initial=await response.json();assert initial['id'].startswith('job-')
        job=await completed_job(client,initial['id'])
        assert job['state']=='completed' and job['run_id']=='run-actual-1' and job['run_id']!=job['id']
        assert executor.calls[0][0]=='write report' and executor.calls[0][1]==str(workspace)
        assert executor.calls[0][2]['model']=='chosen'
        assert job['summary']['usage']['prompt_tokens'] is None
        events=await (await client.get(f'/api/workbench/jobs/{job["id"]}/events?after=0&limit=2')).json()
        assert len(events['events'])==2 and events['next_cursor']==2
        remaining=await (await client.get(f'/api/workbench/jobs/{job["id"]}/events?after=2')).json()
        assert len(remaining['events'])==1 and remaining['next_cursor']==3
        snapshot=await (await client.get('/api/snapshot')).json()
        assert snapshot['events']==events['events']+remaining['events']
        assert (await client.get('/monitor/')).status==200
        assert (await client.get('/style.css')).status==200
        artifacts=await (await client.get(f'/api/workbench/jobs/{job["id"]}/artifacts')).json()
        assert artifacts['files'][0]['path']=='report.txt' and artifacts['files'][0]['available']
        assert artifacts['answers']['work']=='actual handoff'
        preview=await (await client.get(f'/api/workbench/jobs/{job["id"]}/artifacts/file',params={'path':'report.txt'})).json()
        assert preview['content']=='actual archived output\n' and not preview['truncated']
    async with TestClient(TestServer(create_app(workspace,executor=ExecutionStub()))) as client:
        jobs=(await (await client.get('/api/workbench/jobs')).json())['jobs']
        assert len(jobs)==1 and jobs[0]['id']==initial['id'] and jobs[0]['state']=='completed'
        values=await (await client.get(f'/api/workbench/jobs/{initial["id"]}/events')).json()
        assert len(values['events'])==3


async def test_only_one_job_and_cancel_reaches_executor(workspace,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','secret')
    executor=ExecutionStub(asyncio.Event())
    async with TestClient(TestServer(create_app(workspace,executor=executor))) as client:
        first=await client.post('/api/workbench/jobs',json={'task':'long task'},headers=same_origin(client))
        job=await first.json()
        for _ in range(50):
            if executor.calls:break
            await asyncio.sleep(.005)
        busy=await client.post('/api/workbench/jobs',json={'task':'other'},headers=same_origin(client))
        assert busy.status==409
        stopped=await client.post(f'/api/workbench/jobs/{job["id"]}/cancel',json={},headers=same_origin(client))
        assert stopped.status==200 and (await stopped.json())['state']=='cancelled'
        assert executor.cancelled
        assert len(executor.calls)==1


async def test_origin_host_json_and_config_validation(workspace,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','secret')
    executor=ExecutionStub()
    async with TestClient(TestServer(create_app(workspace,executor=executor))) as client:
        assert (await client.post('/api/workbench/jobs',json={'task':'x'})).status==403
        assert (await client.post('/api/workbench/jobs',json={'task':'x'},headers={'Origin':'https://attacker.invalid'})).status==403
        assert (await client.get('/api/workbench/config',headers={'Host':'attacker.invalid'})).status==403
        assert (await client.post('/api/workbench/jobs',data='{}',headers=same_origin(client))).status==415
        for values in ({'api_key':'secret'}, {'api_key_env':'PATH'}, {'output':'/tmp/leak'}, {'harness':'unadapted'}):
            response=await client.post('/api/workbench/jobs',json={'task':'x','config':values},headers=same_origin(client))
            assert response.status==400
        assert not executor.calls
        assert (await client.get('/api/workbench/jobs/missing')).status==404


async def test_artifact_whitelist_links_bounds_and_unicode_preview(workspace,tmp_path,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','secret')
    app=create_app(workspace,executor=ExecutionStub())
    async with TestClient(TestServer(app)) as client:
        job=await (await client.post('/api/workbench/jobs',json={'task':'x'},headers=same_origin(client))).json()
        job=await completed_job(client,job['id']);prefix=f'/api/workbench/jobs/{job["id"]}/artifacts/file'
        for path in ('../summary.json','/etc/passwd','summary.json'):
            assert (await client.get(prefix,params={'path':path})).status==404
        run=Path(job['summary']['output']);target=run/'artifacts/files/report.txt'
        outside=tmp_path/'outside.txt';outside.write_text('private outside')
        target.unlink();target.symlink_to(outside)
        assert (await client.get(prefix,params={'path':'report.txt'})).status==404
        target.unlink();os.link(outside,target)
        assert (await client.get(prefix,params={'path':'report.txt'})).status==404
        target.unlink();target.write_text('界'*50000)
        response=await client.get(prefix,params={'path':'report.txt'});value=await response.json()
        assert value['truncated'] and not value['binary'] and value['content']
        assert value['limit_bytes']==PREVIEW_BYTES
        assert (await client.get(f'/api/workbench/jobs/{job["id"]}/events?limit=9999')).status==400


async def test_failure_redaction_and_interrupted_history(workspace,monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY','never-expose-this-key')
    async def failure(*args,**kwargs):raise ValueError('failure never-expose-this-key')
    app=create_app(workspace,executor=failure)
    async with TestClient(TestServer(app)) as client:
        job=await (await client.post('/api/workbench/jobs',json={'task':'x'},headers=same_origin(client))).json()
        failed=await completed_job(client,job['id'])
        assert failed['state']=='failed' and '[REDACTED]' in failed['error']
        assert 'never-expose-this-key' not in json.dumps(failed)
    path=workspace/'.agentos/workbench'/f'{job["id"]}.json'
    raw=path.read_text();assert 'never-expose-this-key' not in raw
    saved=json.loads(raw);saved['state']='running';path.write_text(json.dumps(saved))
    async with TestClient(TestServer(create_app(workspace,executor=failure))) as client:
        loaded=await (await client.get('/api/workbench/jobs/'+job['id'])).json()
        assert loaded['state']=='failed' and 'not resumed' in loaded['error']


def test_launcher_uses_selected_python_without_installing():
    import sys
    root=(Path(__file__).resolve().parents[1] / "src")
    result=subprocess.run([str(root/'agentctl'),'app','--help'],cwd=root,
                          env={**os.environ,'AGENTOS_PYTHON':sys.executable},capture_output=True,text=True)
    assert result.returncode==0 and '--env-file' in result.stdout and '--command-sandbox' in result.stdout
