
import numpy as np


class LagBaseline:
    """Predict using one historical lag."""

    def __init__(self, feature_name):
        self.feature_name = feature_name

    def predict(self, X):
        values = X[self.feature_name].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Missing or invalid values in {self.feature_name}"
            )

        return np.maximum(values, 0)


class MovingAverageBaseline:
    """Predict using the historical seven-observation average."""

    def predict(self, X):
        values = X["Sales_RollingMean_7"].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                "Missing or invalid values in Sales_RollingMean_7"
            )

        return np.maximum(values, 0)
