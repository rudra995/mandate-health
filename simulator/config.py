"""Configuration loading for the simulator.

One module, one job: read the YAML files and hand back validated structures.
Every other simulator module takes its numbers from here, so the "no magic
numbers in code" rule has exactly one place it can be broken and exactly one
place to check.

The decline taxonomy gets a real validation pass rather than a bare
``yaml.safe_load``. ``config/decline_codes.yaml`` and the ``DeclineCode`` enum
in ``entities`` describe the same set of codes, and Phase 3's retry allocator
reads the YAML while the simulator emits the enum. If those two ever drift, a
code could be terminal in one place and recoverable in the other, which would
silently spend retry budget on unrecoverable declines. The check below makes
that a startup error instead of a quiet bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from simulator.entities import DeclineClass, DeclineCode

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SIMULATOR_CONFIG = REPO_ROOT / "config" / "simulator.yaml"
DEFAULT_DECLINE_CODES = REPO_ROOT / "config" / "decline_codes.yaml"


def load_simulator_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/simulator.yaml``."""
    target = Path(path) if path is not None else DEFAULT_SIMULATOR_CONFIG
    with target.open("r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    _validate_simulator_config(config)
    return config


def _validate_simulator_config(config: dict[str, Any]) -> None:
    """Fail loudly on the config errors that would otherwise skew the world."""
    required = {
        "simulation",
        "population",
        "income",
        "income_day",
        "spend",
        "merchants",
        "mandate",
        "psps",
        "psp_degradation",
        "slots",
        "baseline_slot_policy",
        "pdn",
        "baseline_retry_policy",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"simulator config is missing sections: {sorted(missing)}")

    _check_weights([b["weight"] for b in config["income"]["bands"]], "income.bands")
    _check_weights([s["weight"] for s in config["income_day"]["segments"]], "income_day.segments")
    _check_weights([s["weight"] for s in config["spend"]["segments"]], "spend.segments")
    _check_weights([p["weight"] for p in config["psps"]], "psps")
    _check_weights(list(config["baseline_slot_policy"]["weights"].values()), "baseline_slot_policy")
    _check_weights(list(config["mandate"]["frequency"]["weights"].values()), "mandate.frequency")

    for merchant in config["merchants"]:
        _check_weights(merchant["plan_weights"], f"merchant {merchant['merchant_id']} plan_weights")
        if len(merchant["plans"]) != len(merchant["plan_weights"]):
            raise ValueError(f"merchant {merchant['merchant_id']}: plans/plan_weights length mismatch")

    pdn = config["pdn"]
    for lead in pdn["permitted_lead_hours"]:
        if not pdn["min_lead_hours"] <= lead <= pdn["max_lead_hours"]:
            raise ValueError(f"pdn lead {lead}h falls outside the regulatory window")
        if lead not in pdn["lead_factor"] or lead not in pdn["cancel_hazard"]:
            raise ValueError(f"pdn lead {lead}h has no lead_factor / cancel_hazard entry")

    if config["baseline_retry_policy"]["max_attempts"] > 3:
        raise ValueError("baseline_retry_policy.max_attempts exceeds the NPCI cap of 3")


def _check_weights(weights: list[float], label: str) -> None:
    total = sum(weights)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{label}: weights sum to {total:.6f}, expected 1.0")
    if any(w < 0 for w in weights):
        raise ValueError(f"{label}: negative weight")


@dataclass(frozen=True, slots=True)
class DeclineTaxonomy:
    """The recoverable / terminal split, loaded from YAML and validated."""

    codes: dict[DeclineCode, dict[str, Any]]
    terminal: frozenset[DeclineCode]
    recoverable: frozenset[DeclineCode]

    def is_terminal(self, code: DeclineCode) -> bool:
        return code in self.terminal

    def retry_eligible(self, code: DeclineCode | None) -> bool:
        """A successful attempt (``code is None``) is never retried either."""
        if code is None:
            return False
        return bool(self.codes[code]["retry_eligible"])

    def describe(self, code: DeclineCode) -> str:
        return str(self.codes[code]["description"])


def load_decline_codes(path: str | Path | None = None) -> DeclineTaxonomy:
    """Load and validate ``config/decline_codes.yaml`` against ``DeclineCode``."""
    target = Path(path) if path is not None else DEFAULT_DECLINE_CODES
    with target.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    yaml_codes = set(raw["codes"])
    enum_codes = {c.value for c in DeclineCode}
    if yaml_codes != enum_codes:
        only_yaml = sorted(yaml_codes - enum_codes)
        only_enum = sorted(enum_codes - yaml_codes)
        raise ValueError(
            "decline taxonomy drift between config and DeclineCode enum: "
            f"only in YAML={only_yaml}, only in enum={only_enum}"
        )

    codes: dict[DeclineCode, dict[str, Any]] = {}
    terminal: set[DeclineCode] = set()
    recoverable: set[DeclineCode] = set()

    for name, entry in raw["codes"].items():
        code = DeclineCode(name)
        if entry["code"] != name:
            raise ValueError(f"decline code {name}: 'code' field disagrees with its key")
        cls = DeclineClass(entry["class"])
        # retry_eligible and class must agree. Two fields saying the same thing
        # is a chance for them to disagree, so it is checked rather than trusted.
        expected_eligible = cls is DeclineClass.RECOVERABLE
        if bool(entry["retry_eligible"]) is not expected_eligible:
            raise ValueError(
                f"decline code {name}: class={cls} contradicts retry_eligible={entry['retry_eligible']}"
            )
        codes[code] = entry
        (terminal if cls is DeclineClass.TERMINAL else recoverable).add(code)

    return DeclineTaxonomy(
        codes=codes,
        terminal=frozenset(terminal),
        recoverable=frozenset(recoverable),
    )
