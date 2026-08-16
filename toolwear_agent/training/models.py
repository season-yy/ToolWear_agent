"""P0 深度学习模型定义。"""

from __future__ import annotations

from torch import Tensor, nn


class LightweightCNN1D(nn.Module):
    """支持 1/3/7 通道和可变窗口长度的轻量一维卷积网络。

    三个卷积块逐步扩大感受野并降低时间分辨率，最后使用自适应平均池化。
    因此分类头不依赖固定窗口长度，页面调整窗口参数时无需重建一套硬编码网络。
    """

    def __init__(
        self,
        *,
        input_channels: int,
        base_channels: int = 32,
        class_count: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_channels not in {1, 3, 7}:
            raise ValueError("P0 1D-CNN 仅支持 1、3 或 7 个输入通道。")
        if base_channels < 1:
            raise ValueError("base_channels 必须大于 0。")
        if class_count < 2:
            raise ValueError("class_count 必须大于等于 2。")
        if not 0 <= dropout < 1:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")

        second_channels = base_channels * 2
        third_channels = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(base_channels, second_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(second_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4),
            nn.Conv1d(second_channels, third_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(third_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(third_channels, class_count),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """把形状为 ``[batch, channels, length]`` 的窗口映射为四分类 logits。"""

        if inputs.ndim != 3:
            raise ValueError("1D-CNN 输入必须是 [batch, channels, length] 三维张量。")
        return self.classifier(self.encoder(inputs))
