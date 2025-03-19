# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import unittest
import pytest
import bittensor as bt
import torch

from neurons.validator import Validator
from MIID.base.validator import BaseValidatorNeuron
from MIID.protocol import Dummy, IdentitySynapse
from MIID.utils.uids import get_random_uids
from MIID.validator.reward import get_rewards
from MIID.validator.query_generator import QueryGenerator
from MIID.mock import MockDendrite
from tests.helpers import get_mock_wallet


class TemplateValidatorNeuronTestCase(unittest.TestCase):
    """
    This class contains unit tests for the RewardEvent classes.

    The tests cover different scenarios where completions may or may not be successful and the reward events are checked that they don't contain missing values.
    The `reward` attribute of all RewardEvents is expected to be a float, and the `is_filter_model` attribute is expected to be a boolean.
    """

    def setUp(self):
        sys.argv = sys.argv[0] + ["--config", "tests/configs/validator.json"]

        config = BaseValidatorNeuron.config()
        config.wallet._mock = True
        config.metagraph._mock = True
        config.subtensor._mock = True
        self.neuron = Validator(config)
        self.miner_uids = get_random_uids(self, k=10)

    def test_run_single_step(self):
        # TODO: Test a single step
        pass

    def test_sync_error_if_not_registered(self):
        # TODO: Test that the validator throws an error if it is not registered on metagraph
        pass

    def test_forward(self):
        # TODO: Test that the forward function returns the correct value
        pass

    def test_dummy_responses(self):
        # TODO: Test that the dummy responses are correctly constructed

        responses = self.neuron.dendrite.query(
            # Send the query to miners in the network.
            axons=[
                self.neuron.metagraph.axons[uid] for uid in self.miner_uids
            ],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        for i, response in enumerate(responses):
            self.assertEqual(response, self.neuron.step * 2)

    def test_reward(self):
        # TODO: Test that the reward function returns the correct value
        responses = self.dendrite.query(
            # Send the query to miners in the network.
            axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        rewards = get_rewards(self.neuron, responses)
        expected_rewards = torch.FloatTensor([1.0] * len(responses))
        self.assertEqual(rewards, expected_rewards)

    def test_reward_with_nan(self):
        # TODO: Test that NaN rewards are correctly sanitized
        # TODO: Test that a bt.logging.warning is thrown when a NaN reward is sanitized
        responses = self.dendrite.query(
            # Send the query to miners in the network.
            axons=[self.metagraph.axons[uid] for uid in self.miner_uids],
            # Construct a dummy query.
            synapse=Dummy(dummy_input=self.neuron.step),
            # All responses have the deserialize function called on them before returning.
            deserialize=True,
        )

        rewards = get_rewards(self.neuron, responses)
        expected_rewards = rewards.clone()
        # Add NaN values to rewards
        rewards[0] = float("nan")

        with self.assertLogs(bt.logging, level="WARNING") as cm:
            self.neuron.update_scores(rewards, self.miner_uids)


def test_validator_query_generation():
    """Test that the validator can generate appropriate queries"""
    # Create mock config
    config = bt.config(withconfig=True)
    config.neuron.timeout = 120
    
    # Initialize query generator
    query_generator = QueryGenerator(config)
    
    # Test query generation
    names, template, labels = query_generator.build_queries()
    
    # Assertions
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)
    assert isinstance(template, str)
    assert isinstance(labels, dict)
    assert 'variation_count' in labels


def test_validator_response_handling():
    """Test that the validator can handle different types of responses"""
    # Create mock wallet and dendrite
    wallet = get_mock_wallet()
    mock_dendrite = MockDendrite(wallet)
    
    # Create test synapse
    test_synapse = IdentitySynapse(
        names=["John Smith"],
        query_template="Generate variations for {name}",
        variations={}
    )
    
    # Test valid response
    valid_response = {
        "John Smith": ["Johnny Smith", "J. Smith", "John S."]
    }
    
    # Mock successful response
    response = mock_dendrite.mock_response(test_synapse, valid_response)
    assert isinstance(response.variations, dict)
    assert "John Smith" in response.variations
    
    # Test error response
    error_response = mock_dendrite.mock_error_response(test_synapse)
    assert error_response.dendrite.status_code != 200


def test_validator_reward_calculation():
    """Test the reward calculation logic"""
    # Create test responses
    good_response = {
        "John Smith": ["Johnny Smith", "J. Smith", "John S."]
    }
    empty_response = {}
    invalid_response = {
        "John Smith": []
    }
    
    # Test reward calculations
    from MIID.validator.reward import get_name_variation_rewards
    
    good_score = get_name_variation_rewards(good_response, ["John Smith"])
    empty_score = get_name_variation_rewards(empty_response, ["John Smith"])
    invalid_score = get_name_variation_rewards(invalid_response, ["John Smith"])
    
    assert good_score > empty_score
    assert good_score > invalid_score
    assert 0 <= good_score <= 1
    assert 0 <= empty_score <= 1
    assert 0 <= invalid_score <= 1
