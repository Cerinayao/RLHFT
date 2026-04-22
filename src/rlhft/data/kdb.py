from __future__ import annotations

import time
from typing import Any

import numpy as np

if not hasattr(np, "bool"):
    # `qpython` still expects the removed NumPy alias on newer versions.
    np.bool = np.bool_

from qpython import qconnection

from rlhft.config import KDBConfig


class KDBConnection:
    """Context-managed KDB+ connection with automatic retry."""

    def __init__(self, cfg: KDBConfig) -> None:
        self.cfg = cfg
        self._conn: qconnection.QConnection | None = None

    def __enter__(self) -> KDBConnection:
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def open(self) -> None:
        try:
            self._conn = qconnection.QConnection(
                host=self.cfg.host,
                port=self.cfg.port,
                pandas=False,
            )
            self._conn.open()
        except Exception:
            self._conn = self._connect_with_retry()

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def execute(self, query: str) -> Any:
        try:
            return self._conn(query)
        except Exception:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = self._connect_with_retry()
            return self._conn(query)

    def _connect_with_retry(self) -> qconnection.QConnection:
        while True:
            try:
                q = qconnection.QConnection(
                    host=self.cfg.host,
                    port=self.cfg.retry_port,
                    pandas=False,
                )
                q.open()
                return q
            except Exception as e:
                print(f"Connection failed: {e}")
                time.sleep(self.cfg.retry_interval)
