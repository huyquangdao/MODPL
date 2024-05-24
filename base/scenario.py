from abc import ABC, abstractmethod


class Scenario(ABC):

    def __init__(self, config, datasets, pipelines, evaluators):
        """
        constructor for class abstract class Scenario
        :param config: configuration of the scenario
        :param datasets: set of datasets.
        :param pipelines: set of pipelines.
        :param evaluators: set of evaluators.
        """
        self.config = config
        self.datasets = datasets
        self.pipelines = pipelines
        self.evaluators = evaluators

    @abstractmethod
    def act(self):
        """
        Method that executes pipeline in the set of given datasets.
        :return:
        """
        # loop over the set of datasets.
        for dataset in self.datasets:
            # loop over the set of pipelines.
            for pipeline in self.pipelines:
                pipeline.run(dataset, self.evaluators)


class RecommendationScenario(Scenario):

    def act(self):
        super().act()


class NegotiationScenario(Scenario):

    def act(self):
        super().act()


class ClarificationScenario(Scenario):

    def act(self):
        super().act()
