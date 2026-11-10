import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from bittensor_wallet import Keypair

from MIID.utils.submission_integrity import verify_commitment
from MIID.validator.reward import PARTNER_HOTKEY, BURN_UID, apply_reputation_rewards


def test_existing_miner_commitment_is_valid_and_body_hash_is_bound():
    key = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    challenge = 'challenge_123_test'
    digest = hashlib.sha256(b'synthetic fixture').hexdigest()
    prefix = key.sign(f'{challenge}:{key.ss58_address}'.encode()).hex()[:16]
    fields = dict(challenge_id=challenge, hotkey=key.ss58_address, image_hash=digest,
                  signature=key.sign(f'challenge:{challenge}:hash:{digest}'.encode()).hex(),
                  s3_key=f'submissions/{challenge}/{key.ss58_address}/{prefix}/seed/pose.png.tlock',
                  path_signature=prefix)
    verify_commitment(**fields)
    for altered in ({'image_hash': '0' * 64}, {'challenge_id': 'other'},
                    {'signature': '00' * 64}, {'s3_key': '../../outside'}):
        with pytest.raises(ValueError):
            verify_commitment(**(fields | altered))


@pytest.mark.parametrize('has_uav', [True, False])
@pytest.mark.parametrize('has_kav', [True, False])
def test_reserved_uids_are_unique_and_policy_shares_preserved(has_uav, has_kav):
    hotkeys = [f'hotkey-{uid}' for uid in range(60)]
    hotkeys[14] = PARTNER_HOTKEY
    metagraph = SimpleNamespace(hotkeys=hotkeys, n=60)
    rep = {hk: {'rep_score': 5., 'rep_tier': 'Neutral'} for hk in (hotkeys[1], hotkeys[14], hotkeys[59])} if has_uav else {}
    rewards, uids, metrics = apply_reputation_rewards(
        np.array([1., 1., 1., 1.]) if has_kav else np.zeros(4),
        [1, 14, 59, 1], rep, metagraph)
    assert len(set(uids.tolist())) == len(uids)
    assert sum(rewards) == pytest.approx(1.)
    by_uid = dict(zip(uids.tolist(), rewards.tolist()))
    assert by_uid[14] == pytest.approx(.50 if has_uav else .95)
    assert by_uid.get(BURN_UID, 0.) == pytest.approx(0. if has_kav else .05)
    assert by_uid[1] == pytest.approx((.45 if has_uav else 0.) + (.05 if has_kav else 0.))


@pytest.mark.parametrize('url', ['', 'http://grader.example', 'https://user:pass@grader.example',
                                  'https://grader.example?other=1'])
def test_grading_refuses_insecure_endpoints(monkeypatch, url):
    from MIID.validator import reward
    monkeypatch.setenv('MIID_GRADING_API_URL', url)
    with pytest.raises(ValueError):
        reward._grading_api_urls()


def test_grading_requires_matching_challenge_and_no_redirects(monkeypatch):
    from unittest.mock import Mock, patch
    from MIID.validator import reward
    monkeypatch.setenv('MIID_GRADING_API_URL', 'https://grader.example')
    monkeypatch.setattr(reward, 'GRADING_POLL_INITIAL_WAIT_SECONDS', 0)
    def response(body):
        result = Mock(status_code=200, content=b'json')
        result.json.return_value = body
        return result
    for challenge in ['expected', 'different']:
        with patch.object(reward.requests, 'post', side_effect=[
            response({'status': 'processing'}),
            response({'status': 'completed', 'result': {'challenge_id': challenge, 'results_by_miner': {}}})
        ]) as post:
            if challenge == 'expected':
                result = reward._submit_and_poll_grading_api({'signature': 'fixture'}, 'expected')
                assert result['challenge_id'] == 'expected'
            else:
                with pytest.raises(ValueError):
                    reward._submit_and_poll_grading_api({'signature': 'fixture'}, 'expected')
            for call in post.call_args_list:
                assert call.args[0].startswith('https://')
                assert call.kwargs['allow_redirects'] is False
                assert call.kwargs.get('verify', True) is True
