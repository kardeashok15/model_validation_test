import pandas as pd
from model.direct_deposit import DLSparkUDF, DLConstants
import pkg_resources
import yaml


class NotLoaded(Exception):
    pass


class MyModel(object):

    def __init__(self):
        config = yaml.safe_load(
            pkg_resources.resource_string('model', 'resources/datalake.yaml')
        )
        self._model = DLSparkUDF(config)

    def load_from_s3(self, bucket, key):
        return self

    def load_from_local(self, filepath):
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Generate predictions"""
        if self._model is None:
            raise NotLoaded("""
                model is not yet loaded.
                Initialize model by calling load_from_s3 or load_from_local.
            """)

        def _f(x):
            out = self._model.direct_deposit(x.to_dict())
            if out == DLConstants.DIRECT_DEPOSIT:
                return {'DD': 1.0, 'NOT_DD': 0.0}
            else:
                return {'DD': 0.0, 'NOT_DD': 1.0}

        return pd.Series(data.apply(_f, axis=1))
