from tf_agents.metrics import tf_metrics
from tf_agents.eval.metric_utils import log_metrics
import logging

#params






#metrics
train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.MaxReturnMetric(),
]


class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")



logging.getLogger().setLevel(logging.INFO)
log_metrics(train_metrics)
