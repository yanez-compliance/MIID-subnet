import time
from datetime import datetime, timedelta
from typing import List

import bittensor as bt
import numpy as np
import yfinance as yf
from pytz import timezone

from snp_oracle.predictionnet.protocol import Challenge


################################################################################
#                              Helper Functions                                #
################################################################################
def calc_raw(self, uid, response: Challenge, close_price: float):
    """
    Calculate delta and whether the direction of prediction was correct for one miner over the past N_TIMEPOINTS epochs

    Args:
        uid: The miner uid taken from the metagraph for this response
        response: The synapse response from the miner containing the prediction
        close_price: The S&P500 close price for the current epoch (N_TIMEPOINTS+1 to calculate price direction)

    Returns:
        deltas: The absolute difference between the predicted price and the true price
            - numpy.ndarray: (N_TIMEPOINTS (epochs) x N_TIMEPOINTS (timepoints))
        correct_dirs: A boolean array for if the predicted direction matched the true direction
            - numpy.ndarray: (N_TIMEPOINTS (epochs) x N_TIMEPOINTS (timepoints))

    Notes:
         - first row is the current epoch with only one prediction from <self.prediction_interval> minutes ago
         - the final row is <N_TIMEPOINTS> epochs ago with  (N_TIMEPOINTS x self.prediction_interval = 30min) minute predictions for the current timepoint
         - the final column is the current timepoint with various prediction distances (5min, 10min,...)
    """
    if response.prediction is None:
        return None, None
    elif len(response.prediction) != self.N_TIMEPOINTS:
        return None, None
    else:
        past_predictions = self.past_predictions[uid]
        prediction_array = np.concatenate(
            (
                np.array(response.prediction).reshape(1, self.N_TIMEPOINTS),
                past_predictions,
            ),
            axis=0,
        )
        close_price_array = np.repeat(
            np.array(close_price[1:]).reshape(1, self.N_TIMEPOINTS),
            self.N_TIMEPOINTS,
            axis=0,
        )
        if len(past_predictions.shape) == 1:
            # before_pred_vector = np.array([])
            before_close_vector = np.array([])
        else:
            # add the timepoint before the first t from past history for each epoch
            past_timepoint = close_price[0:-1]
            past_timepoint.reverse()
            before_close_vector = np.array(past_timepoint).reshape(self.N_TIMEPOINTS, 1)
        # take the difference between timepoints and remove the oldest prediction epoch (it is now obselete)
        pred_dir = before_close_vector - prediction_array[:-1, :]
        close_dir = before_close_vector - close_price_array
        correct_dirs = (close_dir >= 0) == time_shift((pred_dir >= 0))
        deltas = np.abs(close_price_array - time_shift(prediction_array[:-1, :]))
        return deltas, correct_dirs


def rank_miners_by_epoch(deltas: np.ndarray, correct_dirs: np.ndarray) -> np.ndarray:
    """
    Generates the rankings for each miner (rows) first according to their correct_dirs (bool), then by deltas (float)

    Args:
        deltas (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint (e.g. the 5min prediction)
        correct_dirs (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint

    Returns:
        all_ranks (numpy.ndarray): n_miners x N_TIMEPOINTS array for one prediction timepoint
    """
    correct_deltas = np.full(deltas.shape, np.nan)
    correct_deltas[correct_dirs] = deltas[correct_dirs]
    incorrect_deltas = np.full(deltas.shape, np.nan)
    incorrect_deltas[~correct_dirs] = deltas[~correct_dirs]
    correct_ranks = rank_columns(correct_deltas)
    incorrect_ranks = rank_columns(incorrect_deltas) + np.nanmax(correct_ranks, axis=0)
    all_ranks = correct_ranks
    all_ranks[~correct_dirs] = incorrect_ranks[~correct_dirs]
    return all_ranks


def rank_columns(array) -> np.ndarray:
    """
    Changes the values of array into within-column ranks, preserving nans

    Args:
        array (numpy.ndarray): a 2D array of values

    Returns:
        ranked_array (numpy.ndarray): array where the values are replaced with within-column rank
    """
    ranked_array = np.copy(array)
    # Iterate over each column
    for col in range(array.shape[1]):
        # Extract the column
        col_data = array[:, col]
        # Get indices of non-NaN values
        non_nan_indices = ~np.isnan(col_data)
        # Extract non-NaN values and sort them
        non_nan_values = col_data[non_nan_indices]
        sorted_indices = np.argsort(non_nan_values)
        ranks = np.empty_like(non_nan_values)
        # Assign ranks
        ranks[sorted_indices] = np.arange(1, len(non_nan_values) + 1)
        # Place ranks back into the original column, preserving NaNs
        ranked_array[non_nan_indices, col] = ranks
    return ranked_array


def time_shift(array) -> np.ndarray:
    """
    This function alligns the timepoints of past_predictions with the current epoch
    and replaces predictions that havent come to fruition with nans.

    Args:
        array (np.ndarray): a square matrix

    Returns:
        shifted_array (np.ndarray): a square matrix where the diagonal elements become the last column,
            the unfulfilled predictions are removed and filled with nans

    Example:
        >>> test_array = np.array([[0,5,10,15,20,25], # - response.prediction on the current timepoint (requested 5 minutes ago)
                                   [-5,0,5,10,15,20], # - 10 minute prediction for time 0
                                   [-10,-5,0,5,10,15],
                                   [-15,-10,-5,0,5,10],
                                   [-20,-15,-10,-5,0,5],
                                   [-25,-20,-15,-10,-5,0], # - 30 minute prediction for time 0
                                   [-30,-25,-20,15,-10,-5]])  # - the obseleted prediction
        >>> shifted_array =time_shift(test_array)
        >>> print(shifted_array)
    """
    shifted_array = np.full((array.shape[0], array.shape[1]), np.nan)
    for i in range(array.shape[0]):
        if i != range(array.shape[0]):
            shifted_array[i, -i - 1 :] = array[i, 0 : i + 1]
        else:
            shifted_array[i, :] = array[i, :]
    return shifted_array


def update_synapse(self, uid, response: Challenge) -> None:
    """
    Updates the values of past_predictions with the current epoch

    Args:
        uid (int): The miner uid taken from the metagraph to be updated
        response (Challenge): The synapse response from the miner containing the prediction

    Returns:
        changes the value of self.past_predictions[uid] to include the most recent prediction and remove the oldest prediction
    """
    past_predictions = self.past_predictions[uid]
    # does not save predictions that mature after market close
    if (
        datetime.now(timezone("America/New_York")).replace(hour=16, minute=5, second=0, microsecond=0)
        - datetime.fromisoformat(response.timestamp)
    ).seconds < self.prediction_interval * 60:
        sec_to_market_close = (
            datetime.now(timezone("America/New_York")).replace(hour=16, minute=0, second=0, microsecond=0)
            - datetime.fromisoformat(response.timestamp)
        ).seconds
        epochs_to_market_close = int((sec_to_market_close / 60) / self.prediction_interval)
        prediction_vector = np.concatenate(
            (
                np.array(response.prediction[0:epochs_to_market_close]),
                (self.N_TIMEPOINTS - epochs_to_market_close) * [np.nan],
            ),
            axis=0,
        )
    else:
        prediction_vector = np.array(response.prediction).reshape(1, self.N_TIMEPOINTS)
    new_past_predictions = np.concatenate((prediction_vector, past_predictions), axis=0)
    self.past_predictions[uid] = new_past_predictions[0:-1, :]  # remove the oldest epoch


################################################################################
#                                Main Function                                 #
################################################################################
def get_rewards(
    self,
    responses: List[Challenge],
    miner_uids: List[int],
) -> np.ndarray:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[Challenge]): A list of responses from the miner.

    Returns:
    - np.ndarray: A tensor of rewards for the given query and responses.
    """
    N_TIMEPOINTS = self.N_TIMEPOINTS
    prediction_interval = self.prediction_interval
    if len(responses) == 0:
        bt.logging.info("Got no responses. Returning reward tensor of zeros.")
        return [], np.full(len(self.metagraph.S), 0.0)  # Fallback strategy: Log and return 0.

    # Prepare to extract close price for this timestamp
    ticker_symbol = "^GSPC"

    timestamp = responses[0].timestamp
    timestamp = datetime.fromisoformat(timestamp)

    # Round up current timestamp and then wait until that time has been hit
    rounded_up_time = (
        timestamp
        - timedelta(
            minutes=timestamp.minute % prediction_interval,
            seconds=timestamp.second,
            microseconds=timestamp.microsecond,
        )
        + timedelta(minutes=prediction_interval, seconds=10)
    )

    ny_timezone = timezone("America/New_York")

    while datetime.now(ny_timezone) < rounded_up_time:
        bt.logging.info(f"Waiting for next {prediction_interval}m interval...")
        if datetime.now(ny_timezone).minute % 10 == 0:
            self.resync_metagraph()
        time.sleep(15)

    prediction_times = []
    rounded_up_time = rounded_up_time.replace(tzinfo=None) - timedelta(seconds=10)
    # add an extra timepoint for dir_acc calculation
    for i in range(N_TIMEPOINTS + 1):
        prediction_times.append(rounded_up_time - timedelta(minutes=(i + 1) * prediction_interval))
    bt.logging.info(f"Prediction times: {prediction_times}")
    data = yf.download(tickers=ticker_symbol, period="5d", interval="5m", progress=False)
    close_price = data.iloc[data.index.tz_localize(None).isin(prediction_times)]["Close"].tolist()
    if len(close_price) < (N_TIMEPOINTS + 1):
        # edge case where its between 9:30am and 10am
        close_price = data.iloc[-N_TIMEPOINTS - 1 :]["Close"].tolist()
    close_price_revealed = " ".join(str(price) for price in close_price)

    bt.logging.info(f"Revealing close prices for this interval: {close_price_revealed}")

    # Preallocate an array (nMiners x N_TIMEPOINTS x N_TIMEPOINTS) where the third dimension is t-1, t-2,...,t-N_TIMEPOINTS for past predictions
    raw_deltas = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
    raw_correct_dir = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), False)
    ranks = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
    for x, response in enumerate(responses):
        # calc_raw also does many helpful things like shifting epoch to
        delta, correct = calc_raw(self, miner_uids[x], response, close_price)
        if delta is None or correct is None:
            if response.prediction is None:
                # no response generated
                bt.logging.info(f"Netuid {x} returned no response. Setting incentive to 0")
                raw_deltas[x, :, :], raw_correct_dir[x, :, :] = np.nan, np.nan
            else:
                # wrong size response generated
                bt.logging.info(
                    f"Netuid {x} returned {len(response.prediction)} predictions instead of {N_TIMEPOINTS}. Setting incentive to 0"
                )
                raw_deltas[x, :, :], raw_correct_dir[x, :, :] = np.nan, np.nan
            continue
        else:
            raw_deltas[x, :, :] = delta
            raw_correct_dir[x, :, :] = correct
        update_synapse(self, miner_uids[x], response)

    # raw_deltas is now a full of the last N_TIMEPOINTS of prediction deltas, same for raw_correct_dir
    ranks = np.full((len(responses), N_TIMEPOINTS, N_TIMEPOINTS), np.nan)
    for t in range(N_TIMEPOINTS):
        ranks[:, :, t] = rank_miners_by_epoch(raw_deltas[:, :, t], raw_correct_dir[:, :, t])

    incentive_ranks = np.nanmean(np.nanmean(ranks, axis=2), axis=1).argsort().argsort()
    reward = np.exp(-0.05 * incentive_ranks)
    reward[incentive_ranks > 100] = 0
    reward = reward / np.max(reward)
    return reward
