from __future__ import annotations

import subprocess
from collections.abc import Iterator


def stream_command(cmd: list[str]) -> tuple[Iterator[str], subprocess.Popen]:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    def _iterator():
        for line in process.stdout:
            yield line

    return _iterator(), process