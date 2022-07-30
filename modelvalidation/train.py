#!/usr/bin/env python
import dill
from metaflow import FlowSpec, step, Parameter


class SubOptimalModel(Exception):
    pass


class TrainModel(FlowSpec):

    pkl_location = Parameter(
        'pkl_location', help="Final Location", default="model.pkl"
    )
    date_key = Parameter(
        'date_key', help="date_key", default='20210307'
    )

    @step
    def start(self):
        self.next(self.train)

    @step
    def train(self):
        with open(self.pkl_location, 'wb') as fp:
            dill.dump({}, fp)

        # Call End Step
        self.next(self.end)

    @step
    def end(self):
        print("done")


if __name__ == '__main__':
    TrainModel()
