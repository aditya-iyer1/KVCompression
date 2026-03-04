"""KVCache-Factory subprocess engine.

Runs KVCache-Factory's LongBench CLI in a subprocess and returns the single-example
prediction text as an EngineResult.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseEngine, EngineResult


def _tail_lines(text: str, *, max_lines: int = 50) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _safe_list_dir(dir_path: Path, *, max_entries: int = 200) -> Dict[str, Any]:
    """Best-effort diagnostic listing (non-recursive)."""
    try:
        entries = sorted(dir_path.iterdir(), key=lambda p: p.name)
    except Exception as e:  # pragma: no cover
        return {"error": f"failed to list {str(dir_path)}: {e!r}"}

    listing = []
    for p in entries[:max_entries]:
        try:
            listing.append(
                {
                    "name": p.name,
                    "is_dir": p.is_dir(),
                    "size": (p.stat().st_size if p.is_file() else None),
                }
            )
        except Exception:
            listing.append({"name": p.name, "is_dir": None, "size": None})
    return {
        "dir": str(dir_path),
        "count": len(entries),
        "truncated": len(entries) > max_entries,
        "entries": listing,
    }


def _find_method_json(save_dir: Path, *, method: str) -> Optional[Path]:
    candidates: List[Path] = []
    target_name = f"{method}.json"

    for root, _, files in os.walk(save_dir):
        for fn in files:
            if fn == target_name:
                candidates.append(Path(root) / fn)

    if not candidates:
        return None

    # Pick the newest file (most recently written)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_records(path: Path) -> List[Dict[str, Any]]:
    """Load KVCache-Factory output file.

    KVCache-Factory writes JSON Lines into a file named *.json; we support:
    - JSON list-of-records
    - JSON dict (single record or container)
    - JSON Lines (one dict per line)
    """
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []

    # Try normal JSON first
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        # If it's a container with a list, use it; else treat dict as a single record
        for k in ("records", "data", "examples", "outputs"):
            v = obj.get(k)
            if isinstance(v, list):
                return [r for r in v if isinstance(r, dict)]
        return [obj]
    return []


def _extract_prediction_text(records: List[Dict[str, Any]]) -> str:
    if not records:
        return ""
    rec = records[0]

    # KVCache-Factory uses "pred" in run_longbench.py.
    for k in ("pred", "prediction", "output", "text", "completion", "answer"):
        v = rec.get(k)
        if isinstance(v, str):
            return v

    # Some wrappers store OpenAI-style payloads.
    choices = rec.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        if isinstance(choice0, dict):
            msg = choice0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(choice0.get("text"), str):
                return choice0["text"]

    # Fallback: stringify something useful
    for k in rec.keys():
        v = rec.get(k)
        if isinstance(v, str):
            return v
    return ""


class KVFactoryEngine(BaseEngine):
    def __init__(
        self,
        kv_factory_repo_dir: str,
        *,
        method: str,
        attn_implementation: str = "sdpa",
        max_capacity_prompts: int = -1,
        max_capacity_prompts_ratio: float = -1.0,
        pruning_ratio: float = -1.0,
        save_root_dir: Optional[str] = None,
    ) -> None:
        self.kv_factory_repo_dir = str(kv_factory_repo_dir)
        self.method = str(method)
        self.attn_implementation = str(attn_implementation)
        self.max_capacity_prompts = int(max_capacity_prompts)
        self.max_capacity_prompts_ratio = float(max_capacity_prompts_ratio)
        self.pruning_ratio = float(pruning_ratio)

        if save_root_dir is None:
            save_root_dir = str(Path("runs") / "kvfactory")
        self.save_root_dir = str(save_root_dir)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EngineResult:
        dataset = kwargs.get("dataset")
        if not dataset or not isinstance(dataset, str):
            raise RuntimeError("KVFactoryEngine requires kwargs['dataset'] (e.g., 'trec' or 'narrativeqa').")

        # Allow per-call overrides.
        method = str(kwargs.get("method", self.method))
        attn_implementation = str(kwargs.get("attn_implementation", self.attn_implementation))
        max_capacity_prompts = int(kwargs.get("max_capacity_prompts", self.max_capacity_prompts))
        max_capacity_prompts_ratio = float(kwargs.get("max_capacity_prompts_ratio", self.max_capacity_prompts_ratio))
        pruning_ratio = float(kwargs.get("pruning_ratio", self.pruning_ratio))

        # Convert OpenAI-style chat messages to a single prompt string.
        prompt_lines: List[str] = []
        for m in messages:
            role = str(m.get("role", ""))
            content = str(m.get("content", ""))
            prompt_lines.append(f"{role.upper()}: {content}")
        prompt = "\n\n".join(prompt_lines).strip() + "\n"

        save_root = Path(self.save_root_dir)
        save_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(
            tempfile.mkdtemp(prefix="kvfactory_", dir=str(save_root))
        )

        # Persist prompt for debugging / reproducibility (KVCache-Factory CLI may not consume it).
        prompt_path = run_dir / "prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        repo_dir = Path(self.kv_factory_repo_dir)
        script_path = repo_dir / "run_longbench.py"
        if not script_path.exists():
            # Keep the temp directory for inspection.
            raise RuntimeError(f"KVCache-Factory script not found at {str(script_path)} (save_dir={str(run_dir)}).")

        cmd: List[str] = [
            sys.executable,
            "-u",
            str(script_path),
            "--dataset",
            dataset,
            "--model_path",
            str(model),
            "--method",
            method,
            "--eval_batch_size",
            "1",
            "--max_num_examples",
            "1",
            "--attn_implementation",
            attn_implementation,
            "--max_capacity_prompts",
            str(max_capacity_prompts),
            "--max_capacity_prompts_ratio",
            str(max_capacity_prompts_ratio),
            "--pruning_ratio",
            str(pruning_ratio),
            "--save_dir",
            str(run_dir),
        ]

        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(repo_dir),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        latency_s = time.time() - start

        stdout_tail = _tail_lines(proc.stdout or "", max_lines=50)
        stderr_tail = _tail_lines(proc.stderr or "", max_lines=50)

        if proc.returncode != 0:
            raise RuntimeError(
                "KVCache-Factory LongBench subprocess failed "
                f"(returncode={proc.returncode}, save_dir={str(run_dir)}).\n\n"
                f"stderr (tail):\n{stderr_tail}"
            )

        out_json = _find_method_json(run_dir, method=method)
        if out_json is None:
            listing = _safe_list_dir(run_dir)
            raise RuntimeError(
                f"KVCache-Factory did not produce {method}.json under save_dir={str(run_dir)}.\n"
                f"dir_listing={listing}\n"
                f"stdout_tail=\n{stdout_tail}\n"
                f"stderr_tail=\n{stderr_tail}"
            )

        try:
            records = _load_records(out_json)
        except Exception as e:
            listing = _safe_list_dir(out_json.parent)
            raise RuntimeError(
                f"Failed to parse KVCache-Factory output JSON at {str(out_json)} (save_dir={str(run_dir)}): {e!r}\n"
                f"parent_dir_listing={listing}\n"
                f"stdout_tail=\n{stdout_tail}\n"
                f"stderr_tail=\n{stderr_tail}"
            ) from e

        prediction_text = _extract_prediction_text(records)
        if not isinstance(prediction_text, str):
            prediction_text = str(prediction_text)

        raw = {
            "kvfactory": {
                "cmd": cmd,
                "returncode": proc.returncode,
                "dataset": dataset,
                "method": method,
                "attn_implementation": attn_implementation,
                "max_capacity_prompts": max_capacity_prompts,
                "max_capacity_prompts_ratio": max_capacity_prompts_ratio,
                "pruning_ratio": pruning_ratio,
                "output_json": str(out_json) if out_json else None,
            },
            "save_dir": str(run_dir),
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }

        return EngineResult(
            text=prediction_text,
            raw=raw,
            finish_reason="stop",
            timings={"latency_s": latency_s},
        )

