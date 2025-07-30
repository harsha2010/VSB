from abc import ABC

from vsb.vsb_types import DistanceMetric
from vsb.workloads.parquet_workload.parquet_workload import ParquetWorkload


class AdversarialBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 128

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Euclidean


class HubSpoke(AdversarialBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "hub-spoke", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 1_000_000

    @staticmethod
    def request_count() -> int:
        return 10_000


class Drift(AdversarialBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "drift", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 1_000_000

    @staticmethod
    def request_count() -> int:
        return 1_000
