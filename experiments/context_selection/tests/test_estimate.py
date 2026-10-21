import asyncio
from experiments.context_selection.tests.test_workflow import fixture_manifest
from experiments.context_selection.common import model_config
from experiments.context_selection.freeze import freeze

def test_estimate_uses_frozen_call_count_and_shared_selectors(tmp_path):
    from experiments.context_selection.estimate import estimate
    async def scenario():
        manifest, config = fixture_manifest(), model_config(stub=True)
        bundle = await freeze(manifest,tmp_path/"plans",config)
        result = estimate(manifest,bundle,split="probe",repeats=3)
        assert result["arms"]["all"]["execution_requests"] == 18
        assert result["arms"]["declared"]["estimated_prompt_tokens"] < result["arms"]["all"]["estimated_prompt_tokens"]
        assert all(r["completion_cap_tokens"] == 18*4096 for r in result["arms"].values())
    asyncio.run(scenario())
