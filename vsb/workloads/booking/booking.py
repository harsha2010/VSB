"""
Cohere-768 dataset - 10M records from en wikipedia, embedded using Cohere
(https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en)
"""

from abc import ABC
from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class Booking(ParquetWorkload):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "booking", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 99_900_000

    @staticmethod
    def request_count() -> int:
        return 100

    @staticmethod
    def dimensions() -> int:
        return 384

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Cosine
