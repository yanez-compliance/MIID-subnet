import random
import bittensor as bt
import numpy as np
from typing import List


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    bt.logging.warning(f"#########################################Metagraph: {metagraph}#########################################")
    bt.logging.warning(f"#########################################Metagraph type: {type(metagraph)}#########################################")
    bt.logging.warning(f"#########################################Metagraph n: {metagraph.n.item()}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons: {metagraph.axons}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons type: {type(metagraph.axons)}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid]: {metagraph.axons[uid]}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid] type: {type(metagraph.axons[uid])}#########################################")
    bt.logging.warning(f"#########################################Metagraph axons[uid] is serving: {metagraph.axons[uid].is_serving}#########################################")
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    candidate_uids = []
    avail_uids = []
    bt.logging.info(f"#########################################Metagraph: {self.metagraph}#########################################")
    bt.logging.info(f"#########################################Metagraph type: {type(self.metagraph)}#########################################")
    bt.logging.info(f"#########################################Metagraph n: {self.metagraph.n.item()}#########################################")
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        uid_is_not_excluded = exclude is None or uid not in exclude
        bt.logging.info(f"#########################################UID: {uid} is available: {uid_is_available}#########################################")
        bt.logging.info(f"#########################################UID: {uid} is not excluded: {uid_is_not_excluded}#########################################")
        if uid_is_available:
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)
    bt.logging.info(f"#########################################Candidate uids: {candidate_uids}#########################################")
    bt.logging.info(f"#########################################Available uids: {avail_uids}#########################################")
    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))
    bt.logging.info(f"#########################################K: {k}#########################################")
    # Check if candidate_uids contain enough for querying, if not grab all avaliable uids
    available_uids = candidate_uids
    bt.logging.info(f"#########################################Available uids: {available_uids}#########################################")
    if len(candidate_uids) < k:
        available_uids += random.sample(
            [uid for uid in avail_uids if uid not in candidate_uids],
            k - len(candidate_uids),
        )
    bt.logging.info(f"#########################################Available uids: {available_uids}#########################################")
    uids = np.array(random.sample(available_uids, k))
    bt.logging.info(f"#########################################Uids: {uids}#########################################")
    return uids
