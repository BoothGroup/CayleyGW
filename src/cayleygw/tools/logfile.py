"""The log file: every stage line and every result, in plain text.

:class:`LogFileHandler` appends to a text file beside the console view. It
opens with the versions, the command and the native threads, then writes one
timestamped line per record. Where the console draws a table, the file writes
every value, including those the console leaves out: every option, the moment
norms, the ``N_q`` history per sector, the discarded weights, every orbital of
a quasiparticle and the reconstruction error of every order.
"""

from __future__ import annotations

import logging
import os

from .._helpers.tools.logfile import _Formatter, _header


class LogFileHandler(logging.FileHandler):
    """Append the package's log records to a file as plain text.

    Args:
        path: File to append to; created if it does not exist.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__(path, mode="a", encoding="utf-8")
        self.setFormatter(_Formatter())
        self.stream.write(_header())
        self.flush()


__all__ = ["LogFileHandler"]
