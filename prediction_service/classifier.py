from __future__ import annotations

from abc import ABC, abstractmethod

from .models import DiseasePrediction, SymptomEntryDTO


class ClassifierStrategy(ABC):

    @abstractmethod
    def classify(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        ...


class RandomForestStrategy(ClassifierStrategy):

    def classify(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        raise NotImplementedError


class KNNStrategy(ClassifierStrategy):

    def classify(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        raise NotImplementedError


class DiseaseClassifier:

    def __init__(self, strategy: ClassifierStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ClassifierStrategy) -> None:
        self._strategy = strategy

    def predict(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        return self._strategy.classify(entries)
