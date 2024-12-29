# ---------------------------------------------------------------------------------
# Portions of this file are derived from gluonts
# - Source: https://github.com/awslabs/gluonts
# - Paper: GluonTS: Probabilistic and Neural Time Series Modeling in Python
# - License: Apache-2.0
#
# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------


from typing import Optional
import numpy as np


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.abs(target - forecast)


def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))

def absolute_scaled_error(
    target,
    forecast,
    seasonal_error
) -> np.ndarray:
    return np.abs(target - forecast) / seasonal_error


def mase(
    target: np.ndarray,
    forecast: np.ndarray,
    seasonal_error: np.ndarray,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """
    mase = absolute_scaled_error(target,forecast,seasonal_error)
    mase = mase.filled(0) 
    return np.mean(mase)



def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(
        np.abs(target - forecast) / (np.abs(target) + np.abs(forecast))
    )


def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * ((Y <= \hat{Y}) - q)|)
    """
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))


def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y < \hat{Y})
    """
    return np.mean(target < forecast)