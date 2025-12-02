import sys
from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]) -> None:
    """Default method for executing shell commands with `pytest` and `sh` package.

    :param command: A list of shell commands as strings.
    """
    try:
        # Use sys.executable to ensure we use the same Python interpreter as the test runner
        sh.Command(sys.executable)(*command, _out=sys.stdout, _err=sys.stderr)
    except sh.ErrorReturnCode as e:
        # Always fail on non-zero exit code
        # Note: stderr might be empty if it was redirected to sys.stderr
        msg = e.stderr.decode().strip()
        if msg:
            pytest.fail(reason=msg)
        else:
            pytest.fail(reason=f"Command failed with exit code {e.exit_code}")
