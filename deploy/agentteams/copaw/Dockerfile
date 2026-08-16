# 在官方 v1.2.2 CoPaw Worker 镜像上修正 Windows 换行和 Team 共享目录解析。
ARG SOURCE_IMAGE=toolwear_agent/agentteams-copaw-worker:v1.2.2
FROM ${SOURCE_IMAGE}

RUN sed -i 's/\r$//' /opt/agentteams/scripts/copaw-worker-entrypoint.sh && \
    find /opt/agentteams/scripts/lib -type f -name '*.sh' -exec sed -i 's/\r$//' {} +

COPY patch_team_storage_scope.py /tmp/patch_team_storage_scope.py
RUN python3 /tmp/patch_team_storage_scope.py && rm /tmp/patch_team_storage_scope.py
