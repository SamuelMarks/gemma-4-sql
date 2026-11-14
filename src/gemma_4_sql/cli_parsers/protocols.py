"""Re-exports of public CLI subparser and parsed command argument protocols."""

from __future__ import annotations

from gemma_4_sql.cli_protocols import (
    AgentArgsProtocol,
    ChatArgsProtocol,
    ETLArgsProtocol,
    EvaluateArgsProtocol,
    FewShotArgsProtocol,
    RagArgsProtocol,
    ServeArgsProtocol,
    SubparserCollectionProtocol,
    TrainArgsProtocol,
)

__all__ = [
    "AgentArgsProtocol",
    "ChatArgsProtocol",
    "ETLArgsProtocol",
    "EvaluateArgsProtocol",
    "FewShotArgsProtocol",
    "RagArgsProtocol",
    "ServeArgsProtocol",
    "SubparserCollectionProtocol",
    "TrainArgsProtocol",
]
