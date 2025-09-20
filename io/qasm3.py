from __future__ import annotations

from pathlib import Path
from typing import Union

from openqasm3 import ast, parse, dumps

ProgramLike = ast.Program


def load_qasm3(source: str) -> ProgramLike:
    """Parse OpenQASM 3 text into an AST, preserving dynamic constructs.

    Parameters
    ----------
    source: str
        The OpenQASM 3 program text.
    """
    return parse(source)


def load_qasm3_file(path: Union[str, Path]) -> ProgramLike:
    return load_qasm3(Path(path).read_text())


def dump_qasm3(program: ProgramLike) -> str:
    """Emit OpenQASM 3 from an AST without lowering dynamic features."""
    return dumps(program)


def dump_qasm3_file(program: ProgramLike, path: Union[str, Path]) -> None:
    Path(path).write_text(dump_qasm3(program))

