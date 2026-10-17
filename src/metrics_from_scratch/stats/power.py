"""Power analysis: how many users an A/B test needs to detect an effect of a given size."""

import math
from dataclasses import dataclass
from statistics import NormalDist

_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class PowerPlan:
    """A two-sided test at significance `alpha` that should detect a real effect with probability `power`.

    All formulas use the normal approximation, which is accurate for the sample sizes A/B tests run at.
    Sample sizes are per arm and rounded up.
    """

    alpha: float = 0.05
    power: float = 0.8

    def __post_init__(self):
        if not (0 < self.alpha < 1 and 0 < self.power < 1):
            raise ValueError(f"alpha and power must be in (0, 1), got {self.alpha} and {self.power}")

    @property
    def _z_alpha(self) -> float:
        return _STANDARD_NORMAL.inv_cdf(1 - self.alpha / 2)

    @property
    def _z_power(self) -> float:
        return _STANDARD_NORMAL.inv_cdf(self.power)

    def n_for_means(self, sd_a: float, sd_b: float, mde: float) -> int:
        """Users per arm to detect a difference in means of `mde` (the minimum detectable effect).

        n = (z_alpha + z_power)^2 * (sd_a^2 + sd_b^2) / mde^2. Halving the effect quadruples the sample.
        """
        _check_effect(mde)
        return math.ceil((self._z_alpha + self._z_power) ** 2 * (sd_a**2 + sd_b**2) / mde**2)

    def power_for_means(self, n: int, sd_a: float, sd_b: float, mde: float) -> float:
        """The chance that a test with `n` users per arm detects a true difference of `mde`."""
        _check_effect(mde)
        standard_error = math.sqrt((sd_a**2 + sd_b**2) / n)
        return _STANDARD_NORMAL.cdf(abs(mde) / standard_error - self._z_alpha)

    def n_for_proportions(self, baseline: float, mde: float) -> int:
        """Users per arm to detect a conversion rate moving from `baseline` to `baseline + mde`.

        Under the null both arms share the pooled rate; under the alternative each arm has its own variance.
        """
        _check_effect(mde)
        treated = baseline + mde
        if not (0 < baseline < 1 and 0 < treated < 1):
            raise ValueError(f"rates must be in (0, 1), got {baseline} and {treated}")
        pooled = (baseline + treated) / 2
        null_spread = self._z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        alternative_spread = self._z_power * math.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
        return math.ceil((null_spread + alternative_spread) ** 2 / mde**2)


def _check_effect(mde: float) -> None:
    if mde == 0:
        raise ValueError("the minimum detectable effect must be non-zero")
