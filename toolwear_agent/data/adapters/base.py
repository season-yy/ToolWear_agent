"""数据集 Adapter 的稳定接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from toolwear_agent.schemas import DatasetInspection


class DatasetAdapter(ABC):
    """把不同原始数据布局转换为统一 DatasetManifest。

    Adapter 只允许读取和体检原始数据，不负责切分、窗口缓存或训练。这样能保证
    数据登记不会意外修改研究数据，也让后续 FastAPI 可以安全复用同一入口。
    """

    adapter_id: str

    @abstractmethod
    def inspect(self, root: Path) -> DatasetInspection:
        """发现数据并返回强类型清单和结构化校验结果。"""

        raise NotImplementedError
