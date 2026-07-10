"""Optional wall-clock energy sampling via macOS ``powermetrics``.

Requires root on macOS. When unavailable, helpers return ``None`` so benchmark
runners stay portable.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional


def powermetrics_available() -> bool:
    return platform.system() == "Darwin" and shutil.which("powermetrics") is not None


@dataclass(frozen=True)
class PowerSample:
    duration_s: float
    source: str
    package_energy_uj: Optional[float] = None
    notes: str = ""


def sample_process_window(
    duration_s: float = 1.0,
    *,
    samplers: str = "cpu_power",
) -> Optional[PowerSample]:
    """Best-effort sample. Returns None when powermetrics cannot run.

    Does not request sudo interactively; if the tool needs privileges and fails,
    callers should treat energy as unavailable.
    """
    if not powermetrics_available():
        return None
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    # -i sample period in ms; -n number of samples.
    period_ms = max(100, int(duration_s * 1000))
    cmd = [
        "powermetrics",
        "--samplers",
        samplers,
        "-i",
        str(period_ms),
        "-n",
        "1",
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5.0, duration_s + 5.0),
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        return PowerSample(
            duration_s=duration_s,
            source="powermetrics",
            notes=f"failed: {exc}",
        )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        return PowerSample(
            duration_s=elapsed,
            source="powermetrics",
            notes=(proc.stderr or proc.stdout or "powermetrics exited non-zero")[:300],
        )
    energy_uj = _parse_energy_uj(proc.stdout)
    return PowerSample(
        duration_s=elapsed,
        source="powermetrics",
        package_energy_uj=energy_uj,
        notes="ok" if energy_uj is not None else "parsed_no_energy_field",
    )


def _parse_energy_uj(text: str) -> Optional[float]:
    """Parse a few common powermetrics energy lines into microjoules."""
    # Examples (locale-dependent):
    #   CPU Power: 1234 mW
    #   Combined Power (CPU + GPU + ANE): 2345 mW
    milliwatts: list[float] = []
    for line in text.splitlines():
        lower = line.lower()
        if "power" not in lower:
            continue
        if "mw" not in lower and "mW" not in line:
            continue
        tokens = line.replace(":", " ").split()
        for i, tok in enumerate(tokens):
            if tok.lower() == "mw" and i > 0:
                try:
                    milliwatts.append(float(tokens[i - 1]))
                except ValueError:
                    continue
    if not milliwatts:
        return None
    # Approximate µJ over 1 second sample: mW * 1000 = µJ/s, times 1s sample.
    return float(max(milliwatts) * 1000.0)
