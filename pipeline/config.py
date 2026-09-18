"""Central configuration. Everything secret comes from environment variables (.env locally,
GitHub Actions secrets in CI, Vercel env vars for the dashboard). Nothing secret lives in code."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

FPL_BASE_URL = "https://fantasy.premierleague.com/api"

FPL_TEAM_ID: int = int(os.getenv("FPL_TEAM_ID", "3964630"))

# Safety Rule 2: protected players are a hard override. Matched case-insensitively against
# players.web_name and second_name. Default per the spec: Bryan Mbeumo.
PROTECTED_PLAYERS: tuple[str, ...] = tuple(
    s.strip().lower() for s in os.getenv("PROTECTED_PLAYERS", "Mbeumo").split(",") if s.strip()
)

# FPL statuses under which a protected player MAY be recommended out (injured / suspended / unavailable).
# 'd' (doubtful) is deliberately NOT included: a knock is not a reason to override the user's protection.
PROTECTED_OVERRIDE_STATUSES: frozenset[str] = frozenset({"i", "s", "u", "n"})

# Transfer rules (confirmed 2026/27 via bootstrap-static game_settings: max_extra_free_transfers=4).
FREE_TRANSFERS_PER_GW = 1
MAX_FREE_TRANSFERS = 5
TRANSFER_HIT_COST = 4

POSITIONS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3

UNDERSTAT_ENABLED: bool = os.getenv("UNDERSTAT_ENABLED", "false").lower() in {"1", "true", "yes"}


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Put it in .env (see .env.example) or export it in the environment."
        )
    return url
