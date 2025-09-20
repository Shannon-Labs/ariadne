from __future__ import annotations

from pathlib import Path
from typing import Union

from openqasm3 import ast, parse, dumps


ProgramLike = ast.Program


def load_qasm3(source: str) -> ProgramLike:
    return parse(source)


def load_qasm3_file(path: Union[str, Path]) -> ProgramLike:
    return load_qasm3(Path(path).read_text())


def dump_qasm3(program: ProgramLike) -> str:
    return dumps(program)

