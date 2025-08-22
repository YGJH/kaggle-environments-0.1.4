#!/usr/bin/env python3
"""Wrapper agent that calls the Nuitka-built binary for submission_vMega.

This module exposes the function `agent(obs, config)` which will run the
compiled binary, send a small JSON payload on stdin and expect a single
integer column on stdout. If the binary is missing, crashes, or times out,
the wrapper falls back to a safe valid move.

Notes:
- This wrapper is intended for local use only. Kaggle kernels disallow
  arbitrary native binaries in submission.py, so this wrapper is not
  suitable to submit to Kaggle unchanged.
"""
import os
import json
import subprocess
import shlex
import random
from typing import Any


_BIN_CANDIDATES = [
    os.path.join(os.getcwd(), 'submission_vMega.bin'),
    os.path.join(os.getcwd(), 'submission_vMega.dist', 'submission_vMega.bin'),
    os.path.join(os.getcwd(), 'submission_vMega.dist', 'submission_vMega'),
    os.path.join(os.getcwd(), 'submission_vMega'),
]


def _find_binary():
    return 'submission_vMega.bin'


def _safe_random_move(board, EMPTY=0):
    valids = [c for c in range(7) if board[c] == EMPTY]
    if not valids:
        return 0
    # prefer center if available
    center = 3
    if center in valids:
        return center
    return random.choice(valids)


def agent(obs: Any, config: Any) -> int:
    """Call the compiled binary to get an action, fallback on error.

    Expects the binary to read a JSON object from stdin with keys:
    {"board": [...], "mark": int, "config": {"rows":6,"columns":7,"inarow":4}}
    and to write a single integer (column) to stdout.
    """
    board = obs['board'] if isinstance(obs, dict) and 'board' in obs else getattr(obs, 'board', None)
    mark = obs['mark'] if isinstance(obs, dict) and 'mark' in obs else getattr(obs, 'mark', None)
    # Normalize config
    if isinstance(config, dict):
        cfg = {k: config.get(k) for k in ('rows', 'columns', 'inarow')}
    else:
        cfg = {'rows': getattr(config, 'rows', None), 'columns': getattr(config, 'columns', None), 'inarow': getattr(config, 'inarow', None)}

    if board is None or mark is None:
        return _safe_random_move(board if board is not None else [0]*7)

    bin_path = _find_binary()
    if not bin_path:
        # binary not found; fallback
        return _safe_random_move(board)

    payload = json.dumps({'board': board, 'mark': mark, 'config': cfg})

    try:
        # run the binary and send JSON via stdin
        proc = subprocess.run([bin_path], input=payload.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1.5)
        out = proc.stdout.decode('utf-8').strip()
        if not out:
            return _safe_random_move(board)
        # take first token
        token = out.split()[0]
        try:
            col = int(token)
        except Exception:
            # maybe the binary printed JSON like {"action": 3}
            try:
                j = json.loads(out)
                col = int(j.get('action', j.get('column', j.get('col', 0))))
            except Exception:
                return _safe_random_move(board)

        # validate
        if 0 <= col < (cfg.get('columns') or 7) and board[col] == 0:
            return int(col)
        # invalid move -> fallback
        return _safe_random_move(board)

    except subprocess.TimeoutExpired:
        # binary hung; kill and fallback
        return _safe_random_move(board)
    except Exception:
        return _safe_random_move(board)


if __name__ == '__main__':
    # quick local smoke test
    sample_board = [0]*42
    print('action=', agent({'board': sample_board, 'mark': 1}, {'rows':6,'columns':7,'inarow':4}))
