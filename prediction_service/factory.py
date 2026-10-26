from __future__ import annotations

from abc import ABC, abstractmethod

from .classifier import ClassifierStrategy, KNNStrategy, RandomForestStrategy


class ModelFactory(ABC):

    @abstractmethod
    def create_classifier(self) -> ClassifierStrategy:        ...


class RandomForestFactory(ModelFactory):
    def create_classifier(self) -> ClassifierStrategy:
        return RandomForestStrategy()


class KNNFactory(ModelFactory):
    def create_classifier(self) -> ClassifierStrategy:
        return KNNStrategy()
