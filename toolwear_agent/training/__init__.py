"""训练与评估模块。

后续这里会逐步加入：
- PHM2010 数据读取
- VB 标签生成
- 特征提取
- 小样本训练
- 模型评估和可视化
"""

from toolwear_agent.training.models import LightweightCNN1D
from toolwear_agent.training.service import TrainingService

__all__ = ["LightweightCNN1D", "TrainingService"]
