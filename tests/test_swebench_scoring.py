"""Synthetic official reports and temporary Git repositories; no Docker or LLM."""
import copy
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agentos.cli import main
from agentos.execution.bench import report, run_suite
from agentos.execution.records import inventory
from agentos.execution import swebench as swe


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path/'repo'; repo.mkdir()
    (repo/'change.py').write_text('before\n')
    (repo/'delete.txt').write_text('delete me\n')
    swe.git(repo, 'init', '-q')
    swe.git(repo, 'add', '.')
    swe.git(repo, '-c', 'user.name=test', '-c', 'user.email=test@local', 'commit', '-qm', 'baseline')
    commit = swe.git(repo, 'rev-parse', 'HEAD').decode().strip()
    dataset = tmp_path/'official-data.json'
    dataset.write_text(json.dumps([{'instance_id': 'example__repo-1', 'repo': 'example/repo',
                                    'base_commit': commit, 'test_patch': 'OFFICIAL_TESTS_NOT_MODEL_INPUT'}]))
    spec = {'type': 'swebench', 'python': sys.executable, 'dataset_name': str(dataset), 'base_repo': str(repo),
            'instance_id': 'example__repo-1', 'base_commit': commit, 'timeout': 30}
    return repo, spec


def archived_run(root, repo, spec, *, changes=True, workflow='completed', patch=False):
    run = root/'run-original'; run.mkdir(parents=True)
    starting = inventory(repo)
    deltas = []
    if changes:
        for name, content in [('change.py', b'after\n'), ('added.txt', b'new\n'), ('delete.txt', None)]:
            deltas.append({'path': name, 'before_sha256': starting.get(name),
                           'after_sha256': swe.digest(content) if content is not None else None})
            if content is not None:
                path=run/'artifacts/files'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    summary = {'schema': 'agentos.run.v1', 'run_id': run.name, 'workflow_status': workflow,
               'config': {'profile': 'runtime', 'api_key_env': 'CUSTOM_MODEL_AUTH'},
               'execution_seconds': 9.25, 'evaluation_seconds': 0, 'artifacts': deltas,
               'score': {'status': 'not_requested', 'passed': None},
               'usage': {'prompt_tokens': None, 'completion_tokens': None, 'usage_complete': False,
                         'known_prompt_tokens': 40, 'missing_usage_calls': 1, 'retries': 1},
               'backend_injected_for_test': True}
    (run/'summary.json').write_text(json.dumps(summary))
    (run/'start.json').write_text(json.dumps({'git': {'head': spec['base_commit'], 'status': ''}, 'files': starting}))
    (run/'artifacts.json').write_text(json.dumps({'changes': deltas}))
    (run/'task.json').write_text(json.dumps({'run_id': run.name, 'task': 'make requested changes'}))
    (run/'events.jsonl').write_text('original events\n')
    if patch:
        _,_,data,_=swe.bind_archived_run(run, swe.prepare_spec(spec))
        (run/'model.patch').write_bytes(data)
    return run


def file_hashes(root):
    return {str(p.relative_to(root)): swe.digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}


def fake_official(monkeypatch, *, resolved=True, observed=None):
    async def fake(argv, directory, timeout, *, credential_env=None):
        prediction=json.loads((directory/'predictions.jsonl').read_text())
        run_id=argv[argv.index('--run_id')+1]
        instance=prediction['instance_id']
        log=directory/'logs/run_evaluation'/run_id/prediction['model_name_or_path']/instance
        log.mkdir(parents=True)
        (log/'report.json').write_text(json.dumps({instance:{'patch_exists':True,'patch_successfully_applied':True,'resolved':resolved}}))
        (log/'run_instance.log').write_text('>>>>> Applied Patch\n')
        (directory/'stdout.txt').write_text('synthetic official process\n')
        (directory/'stderr.txt').write_text('')
        if observed is not None:
            observed.append({'argv':argv,'prediction':prediction,'credential_env':credential_env,
                             'dataset':json.loads((directory/'dataset.json').read_text())})
        return {'returncode':0,'timed_out':False}
    monkeypatch.setattr(swe,'run_official',fake)


@pytest.mark.parametrize('kwargs,outcome,passed', [
    ({'instance_report':{'i':{'resolved':True}}}, 'resolved', True),
    ({'instance_report':{'i':{'resolved':False,'patch_successfully_applied':True}}}, 'unresolved', False),
    ({'log':'2026-01-01 - INFO - >>>>> Patch Apply Failed:\nfailed hunk'}, 'patch_apply_error', False),
    ({'aggregate':{'error_ids':['i']},'process':{'returncode':0}}, 'evaluation_error', None),
    ({'log':'BuildImageError: image build failed'}, 'environment_error', None),
    ({'log':'docker.errors.ImageNotFound: not found'}, 'environment_error', None),
    ({'process':{'timed_out':True,'returncode':-9}}, 'evaluation_error', None),
    ({'process':{'returncode':0}}, 'evaluation_error', None),
    ({'empty':True}, 'not_evaluated', None),
    ({'instance_report':{'i':{'patch_successfully_applied':False,'resolved':False}}}, 'evaluation_error', None),
    ({'instance_report':{'i':{'resolved':True}},'log':'+ >>>>> Patch Apply Failed: string inside a patch'}, 'resolved', True),
])
def test_report_classification(kwargs,outcome,passed):
    result=swe.parse_outcome('i',**kwargs)
    assert result['outcome']==outcome and result['passed'] is passed


def test_snapshot_patch_add_delete_binary_and_no_change(fixture,tmp_path):
    repo,spec=fixture
    before=tmp_path/'before';after=tmp_path/'after'
    shutil.copytree(repo,before,ignore=shutil.ignore_patterns('.git'))
    shutil.copytree(before,after)
    assert swe.snapshot_patch(before,after)==b''
    (after/'change.py').write_text('changed\n')
    (after/'delete.txt').unlink()
    (after/'new.bin').write_bytes(b'\x00\xffbinary\x00')
    patch=swe.snapshot_patch(before,after)
    assert b'a/change.py' in patch and b'deleted file mode' in patch and b'GIT binary patch' in patch
    swe.git(before,'init','-q');swe.git(before,'apply','--binary','-',input_data=patch)
    assert inventory(before)==inventory(after)
    assert not (after/'.git').exists()


def test_archive_binding_rejects_tampering_and_wrong_base(fixture,tmp_path):
    repo,spec=fixture;run=archived_run(tmp_path/'records',repo,spec,patch=True)
    frozen=swe.prepare_spec(spec)
    source,raw,patch,binding=swe.bind_archived_run(run,frozen)
    assert binding['patch_source']=='verified_archived_model.patch'
    assert binding['patch_sha256']==swe.digest(patch)
    assert b'new file mode' in patch and b'deleted file mode' in patch
    (run/'artifacts/files/change.py').write_text('tampered')
    with pytest.raises(ValueError,match='artifact hash mismatch'):swe.bind_archived_run(run,frozen)
    (run/'artifacts/files/change.py').write_text('after\n')
    (run/'model.patch').write_text('')
    with pytest.raises(ValueError,match='does not reproduce'):swe.bind_archived_run(run,frozen)
    wrong=copy.deepcopy(spec);wrong['base_commit']='0'*40
    with pytest.raises(ValueError):swe.prepare_spec(wrong)


async def test_append_only_evaluation_and_csv_same_run_metrics(fixture,tmp_path,monkeypatch):
    repo,spec=fixture;run=archived_run(tmp_path/'records',repo,spec,workflow='failed',patch=True)
    before=file_hashes(run);observed=[];fake_official(monkeypatch,observed=observed)
    first=await swe.evaluate_run(run,spec,output=tmp_path/'evals')
    second=await swe.evaluate_run(run,spec,output=tmp_path/'evals')
    assert first['evaluation_id']!=second['evaluation_id']
    assert file_hashes(run)==before
    assert first['score']['outcome']=='resolved'
    row=next(csv.DictReader(io.StringIO(report(first['output']))))
    assert row['run_id']==run.name and row['source_run_id']==run.name
    assert row['workflow_status']=='failed' and row['passed']=='True'
    assert row['execution_seconds']=='9.25' and row['prompt_tokens']=='' and row['usage_complete']=='False'
    assert float(row['evaluation_seconds'])>=0
    assert observed[0]['credential_env']=='CUSTOM_MODEL_AUTH'
    assert observed[0]['prediction']['model_patch']==(run/'model.patch').read_text()
    (Path(first['output'])/'source-summary.json').write_text('{}')
    with pytest.raises(ValueError,match='binding mismatch'):report(first['output'])


async def test_empty_patch_never_invokes_harness(fixture,tmp_path,monkeypatch):
    repo,spec=fixture;run=archived_run(tmp_path/'records',repo,spec,changes=False)
    async def forbidden(*args,**kwargs):raise AssertionError('official harness must not run')
    monkeypatch.setattr(swe,'run_official',forbidden)
    result=await swe.evaluate_run(run,spec,output=tmp_path/'evals')
    assert result['score']['outcome']=='not_evaluated'
    assert result['process']['not_started']=='empty_patch'


async def test_spec_dataset_and_baseline_are_frozen_before_candidate(fixture,tmp_path,monkeypatch):
    repo,spec=fixture;run=archived_run(tmp_path/'records',repo,spec)
    snapshot=tmp_path/'snapshot';shutil.copytree(repo,snapshot,ignore=shutil.ignore_patterns('.git'))
    frozen=swe.prepare_spec(spec,snapshot=snapshot)
    spec['instance_id']='attacker__other-9'
    Path(spec['dataset_name']).write_text('[]')
    observed=[];fake_official(monkeypatch,observed=observed)
    result=await swe.evaluate_run(run,frozen,output=tmp_path/'evals')
    assert result['instance_id']=='example__repo-1'
    assert observed[0]['dataset'][0]['test_patch']=='OFFICIAL_TESTS_NOT_MODEL_INPUT'
    assert not (snapshot/'dataset.json').exists()
    (snapshot/'change.py').write_text('modified baseline')
    restored={**frozen.config};restored.pop('frozen_dataset_sha256');restored.pop('dataset_source_sha256')
    Path(restored['dataset_name']).write_bytes(frozen.dataset_json)
    with pytest.raises(ValueError,match='snapshot'):swe.prepare_spec(restored,snapshot=snapshot)


async def test_manifest_uses_archived_snapshot_patch_and_frozen_policy(fixture,tmp_path,monkeypatch):
    repo,spec=fixture;observed=[];fake_official(monkeypatch,observed=observed)
    manifest=tmp_path/'suite.json';manifest.write_text(json.dumps({'schema':'agentos.bench.v1','tasks':[
        {'id':'one','cwd':'repo','task':'change file','score':spec}]}))
    async def fake_execute(task,*,cwd,overrides,provenance,**kwargs):
        assert not (cwd/'.git').exists()
        assert 'OFFICIAL_TESTS_NOT_MODEL_INPUT' not in ''.join(p.read_text() for p in cwd.iterdir() if p.is_file())
        # Candidate modification of on-disk scoring input must not affect frozen inputs.
        manifest.write_text('{}');Path(spec['dataset_name']).write_text('[]')
        run=archived_run(Path(overrides['output']),repo,spec)
        start=json.loads((run/'start.json').read_text());start['git']={'head':'parent-worktree-head'};start['provenance']=provenance
        (run/'start.json').write_text(json.dumps(start))
        result=json.loads((run/'summary.json').read_text());result['output']=str(run);return result
    monkeypatch.setattr('agentos.execution.bench.execute',fake_execute)
    result=await run_suite(str(manifest),compare='runtime',output=str(tmp_path/'batches'),
                           overrides={'model':'test','base_url':'http://unused.test','command_sandbox':'none'})
    assert len(observed)==1 and observed[0]['dataset'][0]['instance_id']=='example__repo-1'
    assert 'new file mode' in observed[0]['prediction']['model_patch']
    row=next(csv.DictReader(io.StringIO(report(result['output']))))
    assert row['outcome']=='resolved' and row['execution_seconds']=='9.25'


def test_cli_evaluate_has_no_model_path(fixture,tmp_path,monkeypatch,capsys):
    repo,spec=fixture;run=archived_run(tmp_path/'records',repo,spec)
    path=tmp_path/'score.json';path.write_text(json.dumps(spec));fake_official(monkeypatch)
    async def forbidden(*args,**kwargs):raise AssertionError('model execution forbidden')
    monkeypatch.setattr('agentos.execution.service.execute',forbidden)
    assert main(['bench','evaluate','--run',str(run),'--spec',str(path),'--output',str(tmp_path/'evals')])==0
    assert json.loads(capsys.readouterr().out)['score']['outcome']=='resolved'


def test_trusted_environment_strips_arbitrary_model_credential_name(monkeypatch):
    monkeypatch.setenv('CUSTOM_MODEL_AUTH','secret');monkeypatch.setenv('DEEPSEEK_API_KEY','secret')
    assert 'CUSTOM_MODEL_AUTH' not in swe.trusted_env(['CUSTOM_MODEL_AUTH'])
    assert 'DEEPSEEK_API_KEY' not in swe.trusted_env()
