"""只基于 validation 特征生成可复现的 t-SNE 证据。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sample_indices(labels: np.ndarray, *, max_samples: int, random_seed: int) -> np.ndarray:
    """按类别比例抽样，避免大验证集让 t-SNE 成为训练瓶颈。"""

    if labels.size <= max_samples:
        return np.arange(labels.size)
    rng = np.random.default_rng(random_seed)
    selected: list[np.ndarray] = []
    for class_id in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_id)
        quota = max(1, round(max_samples * class_indices.size / labels.size))
        selected.append(rng.choice(class_indices, size=min(quota, class_indices.size), replace=False))
    merged = np.concatenate(selected)
    if merged.size > max_samples:
        merged = rng.choice(merged, size=max_samples, replace=False)
    return np.sort(merged)


def write_validation_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_labels: tuple[str, ...],
    output_file: Path,
    *,
    random_seed: int,
    max_samples: int = 1600,
) -> Path:
    """生成不含 final test 的 validation t-SNE 图。"""

    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    feature_array = np.asarray(features, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    if feature_array.ndim != 2 or feature_array.shape[0] != label_array.size:
        raise ValueError("validation 特征与标签形状不一致。")
    if label_array.size < 2:
        raise ValueError("生成 t-SNE 至少需要 2 个 validation 样本。")

    indices = _sample_indices(label_array, max_samples=max_samples, random_seed=random_seed)
    sampled_features = feature_array[indices]
    sampled_labels = label_array[indices]
    normalized = StandardScaler().fit_transform(sampled_features)
    component_count = min(30, normalized.shape[1], max(1, normalized.shape[0] - 1))
    reduced = PCA(n_components=component_count, random_state=random_seed).fit_transform(normalized)

    if reduced.shape[0] >= 4:
        perplexity = min(30.0, max(2.0, (reduced.shape[0] - 1) / 3.0))
        coordinates = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=500,
            random_state=random_seed,
        ).fit_transform(reduced)
    else:
        # 极小测试夹具无法稳定运行 t-SNE，使用同一 validation PCA 坐标验证证据链。
        coordinates = np.column_stack(
            (reduced[:, 0], reduced[:, 1] if reduced.shape[1] > 1 else np.zeros(reduced.shape[0]))
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.4, 5.4))
    colors = ("#2878FF", "#1FA678", "#F29A00", "#EF6645")
    for class_id, class_name in enumerate(class_labels):
        mask = sampled_labels == class_id
        if not np.any(mask):
            continue
        axis.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=18,
            alpha=0.72,
            color=colors[class_id % len(colors)],
            label=class_name,
        )
    axis.set_title("Validation t-SNE 特征分布（不含 final test）")
    axis.set_xlabel("t-SNE 1")
    axis.set_ylabel("t-SNE 2")
    axis.grid(alpha=0.18)
    axis.legend(loc="best", frameon=True)
    figure.tight_layout()
    figure.savefig(output_file, dpi=180)
    plt.close(figure)
    return output_file
