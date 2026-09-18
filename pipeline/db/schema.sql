-- Vantage: FPL AI assistant schema (Postgres / Supabase).
-- Idempotent: safe to re-run. Raw ingested tables (teams, players, fixtures, player_gw_stats,
-- user_squad, user_entry_state) are kept separate from derived tables (predictions, model_runs,
-- recommendations) so models can always be retrained from scratch against history.

CREATE TABLE IF NOT EXISTS teams (
    id                    INTEGER PRIMARY KEY,          -- FPL team id (1..20, per season)
    season                TEXT NOT NULL,                -- e.g. '2026/27'
    code                  INTEGER,                      -- stable cross-season club code
    name                  TEXT NOT NULL,
    short_name            TEXT,
    strength              INTEGER,
    strength_overall_home INTEGER,
    strength_overall_away INTEGER,
    strength_attack_home  INTEGER,
    strength_attack_away  INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER,
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS players (
    id                            INTEGER PRIMARY KEY,   -- FPL element id (per season)
    season                        TEXT NOT NULL,
    code                          INTEGER,               -- stable cross-season player code
    web_name                      TEXT NOT NULL,
    first_name                    TEXT,
    second_name                   TEXT,
    team_id                       INTEGER REFERENCES teams(id),
    position                      TEXT NOT NULL CHECK (position IN ('GKP','DEF','MID','FWD')),
    price                         NUMERIC(5,1) NOT NULL, -- in £m (now_cost / 10)
    cost_change_start             NUMERIC(5,1),          -- £m change since season start
    status                        TEXT,                  -- a=available, i=injured, s=suspended, d=doubtful, u=unavailable, n=not in squad
    news                          TEXT,
    news_added                    TIMESTAMPTZ,
    chance_of_playing_next_round  INTEGER,               -- 0..100 or NULL (= 100)
    chance_of_playing_this_round  INTEGER,
    selected_by_percent           NUMERIC(6,2),
    total_points                  INTEGER,
    form                          NUMERIC(6,2),
    updated_at                    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS players_team_idx ON players(team_id);
CREATE INDEX IF NOT EXISTS players_code_idx ON players(code);

CREATE TABLE IF NOT EXISTS fixtures (
    id               INTEGER PRIMARY KEY,               -- FPL fixture id
    season           TEXT NOT NULL,
    code             INTEGER,
    gw               INTEGER,                            -- NULL until the fixture is scheduled into an event
    home_team_id     INTEGER REFERENCES teams(id),
    away_team_id     INTEGER REFERENCES teams(id),
    kickoff_time     TIMESTAMPTZ,
    difficulty_home  INTEGER,                            -- FPL FDR faced by the home side
    difficulty_away  INTEGER,
    finished         BOOLEAN NOT NULL DEFAULT false,
    started          BOOLEAN NOT NULL DEFAULT false,
    home_score       INTEGER,
    away_score       INTEGER,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fixtures_gw_idx ON fixtures(gw);

-- One row per player per fixture actually played/scheduled in (a double gameweek gives 2 rows for the same gw).
CREATE TABLE IF NOT EXISTS player_gw_stats (
    player_id                 INTEGER NOT NULL REFERENCES players(id),
    season                    TEXT NOT NULL,
    gw                        INTEGER NOT NULL,
    fixture_id                INTEGER NOT NULL,
    opponent_team_id          INTEGER,
    was_home                  BOOLEAN,
    kickoff_time              TIMESTAMPTZ,
    minutes                   INTEGER NOT NULL DEFAULT 0,
    points                    INTEGER NOT NULL DEFAULT 0,
    goals                     INTEGER NOT NULL DEFAULT 0,
    assists                   INTEGER NOT NULL DEFAULT 0,
    clean_sheet               INTEGER NOT NULL DEFAULT 0,
    goals_conceded            INTEGER NOT NULL DEFAULT 0,
    own_goals                 INTEGER NOT NULL DEFAULT 0,
    penalties_saved           INTEGER NOT NULL DEFAULT 0,
    penalties_missed          INTEGER NOT NULL DEFAULT 0,
    yellow_cards              INTEGER NOT NULL DEFAULT 0,
    red_cards                 INTEGER NOT NULL DEFAULT 0,
    saves                     INTEGER NOT NULL DEFAULT 0,
    bonus                     INTEGER NOT NULL DEFAULT 0,
    bps                       INTEGER NOT NULL DEFAULT 0,
    defensive_contribution    INTEGER NOT NULL DEFAULT 0,
    starts                    INTEGER NOT NULL DEFAULT 0,
    influence                 NUMERIC(7,1),
    creativity                NUMERIC(7,1),
    threat                    NUMERIC(7,1),
    ict_index                 NUMERIC(7,1),
    xg                        NUMERIC(6,2),               -- FPL's own expected_goals
    xa                        NUMERIC(6,2),               -- FPL's own expected_assists
    xgi                       NUMERIC(6,2),
    xgc                       NUMERIC(6,2),               -- expected goals conceded
    value                     NUMERIC(5,1),               -- price at the time (£m)
    selected                  INTEGER,
    transfers_in              INTEGER,
    transfers_out             INTEGER,
    team_h_score              INTEGER,
    team_a_score              INTEGER,
    updated_at                TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, fixture_id)
);
CREATE INDEX IF NOT EXISTS pgs_gw_idx ON player_gw_stats(season, gw);
CREATE INDEX IF NOT EXISTS pgs_player_idx ON player_gw_stats(player_id, gw);

-- Optional supplementary xG/xA from Understat (per player per match). Empty if scraping is disabled/fails.
CREATE TABLE IF NOT EXISTS understat_player_match (
    understat_player_id  INTEGER NOT NULL,
    player_name          TEXT,
    match_date           DATE NOT NULL,
    team                 TEXT,
    opponent             TEXT,
    was_home             BOOLEAN,
    minutes              INTEGER,
    xg                   NUMERIC(6,3),
    xa                   NUMERIC(6,3),
    npxg                 NUMERIC(6,3),
    shots                INTEGER,
    key_passes           INTEGER,
    fpl_player_code      INTEGER,        -- resolved mapping to players.code when possible
    PRIMARY KEY (understat_player_id, match_date)
);

-- The user's actual 15 for a given gw (from /entry/{id}/event/{gw}/picks/).
CREATE TABLE IF NOT EXISTS user_squad (
    season           TEXT NOT NULL,
    gw               INTEGER NOT NULL,
    player_id        INTEGER NOT NULL REFERENCES players(id),
    is_starting      BOOLEAN NOT NULL,
    is_captain       BOOLEAN NOT NULL DEFAULT false,
    is_vice_captain  BOOLEAN NOT NULL DEFAULT false,
    bench_order      INTEGER,                    -- 1..4 for bench, NULL for starters
    squad_position   INTEGER,                    -- FPL pick position 1..15
    purchase_price   NUMERIC(5,1),               -- only known via authenticated my-team endpoint; NULL otherwise
    PRIMARY KEY (season, gw, player_id)
);

-- Snapshot of the user's entry per gw: bank, value, transfers, chips (planning only, Safety Rule 3).
CREATE TABLE IF NOT EXISTS user_entry_state (
    season                 TEXT NOT NULL,
    gw                     INTEGER NOT NULL,
    bank                   NUMERIC(5,1),                 -- £m
    squad_value            NUMERIC(5,1),                 -- £m (selling value)
    event_transfers        INTEGER,
    event_transfers_cost   INTEGER,
    free_transfers_after   INTEGER,                      -- derived: FTs available going INTO gw+1
    active_chip            TEXT,                         -- wildcard / freehit / bboost / 3xc / NULL
    chips_used_json        JSONB,                        -- [{"name":"wildcard","event":12}, ...]
    points                 INTEGER,
    total_points           INTEGER,
    overall_rank           INTEGER,
    gw_rank                INTEGER,
    points_on_bench        INTEGER,
    updated_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, gw)
);

-- ---------- derived ----------

CREATE TABLE IF NOT EXISTS model_runs (
    id             SERIAL PRIMARY KEY,
    run_date       TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version  TEXT NOT NULL,
    position       TEXT,                                 -- GKP/DEF/MID/FWD or 'ALL'
    mae            NUMERIC(8,4),
    rmse           NUMERIC(8,4),
    n_train        INTEGER,
    n_valid        INTEGER,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    player_id              INTEGER NOT NULL REFERENCES players(id),
    season                 TEXT NOT NULL,
    gw                     INTEGER NOT NULL,
    predicted_points       NUMERIC(6,2) NOT NULL,
    predicted_points_low   NUMERIC(6,2),
    predicted_points_high  NUMERIC(6,2),
    model_version          TEXT NOT NULL,
    predicted_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (player_id, season, gw, model_version)
);
CREATE INDEX IF NOT EXISTS predictions_gw_idx ON predictions(season, gw);

CREATE TABLE IF NOT EXISTS recommendations (
    id                     SERIAL PRIMARY KEY,
    season                 TEXT NOT NULL,
    gw                     INTEGER NOT NULL,              -- the gw this recommendation is FOR
    run_date               TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version          TEXT,
    squad_json             JSONB NOT NULL,                -- 15 players with starting/bench/captain flags
    transfers_json         JSONB NOT NULL,                -- [{"out":id,"in":id,"cost_delta":...}], plus hit cost
    captain_id             INTEGER,
    vice_captain_id        INTEGER,
    chip_used              TEXT,                          -- chip recommended for THIS gw, or NULL
    chip_plan_json         JSONB,                         -- per-chip best week + expected gain over horizon
    expected_points_gain   NUMERIC(7,2),                  -- vs. no-transfer baseline over the horizon
    expected_points_gw     NUMERIC(7,2),                  -- predicted XI points this gw (captain doubled)
    warnings_json          JSONB,                         -- e.g. protected-player warnings (Safety Rule 2)
    horizon_weights        TEXT,
    status                 TEXT NOT NULL DEFAULT 'recommended'  -- recommended | user_confirmed | rejected. NEVER auto-submitted.
);
CREATE INDEX IF NOT EXISTS recommendations_gw_idx ON recommendations(season, gw, run_date DESC);

-- Prediction accuracy log: filled once actuals are known.
CREATE TABLE IF NOT EXISTS prediction_accuracy (
    season          TEXT NOT NULL,
    gw              INTEGER NOT NULL,
    model_version   TEXT NOT NULL,
    position        TEXT NOT NULL,
    n               INTEGER,
    mae             NUMERIC(8,4),
    rmse            NUMERIC(8,4),
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (season, gw, model_version, position)
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id           SERIAL PRIMARY KEY,
    run_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    season       TEXT,
    current_gw   INTEGER,
    n_players    INTEGER,
    n_teams      INTEGER,
    n_fixtures   INTEGER,
    n_gw_stats   INTEGER,
    n_understat  INTEGER,
    notes        TEXT
);
