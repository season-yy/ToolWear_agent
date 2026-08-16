"""业务工作流可稳定映射到 Tool API 的错误。"""

from toolwear_agent.core.errors import ToolWearError


class InvalidWorkflowStateError(ToolWearError):
    """动作与实验当前状态不匹配。"""

    error_code = "INVALID_WORKFLOW_STATE"


class TrainingCancelledError(Exception):
    """Worker 在安全检查点发现用户取消请求。"""
