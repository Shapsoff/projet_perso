"""
schema.py — Schéma SQLite de la Player DB
Phase 6 — Bot Poker Académique

Choix de persistance (décision session 6) :
    SQLite en local, PRAGMA journal_mode=WAL. Justification (vitesse avant
    tout, cf. brainstorming session 6) : pas de latence réseau (contrainte
    dure — une décision se prend en quelques ms pendant une main), pas de
    parsing complet d'un fichier à chaque écriture (contrairement à un
    JSON global), requêtable nativement (agrégations, filtrage par joueur),
    zéro dépendance externe (sqlite3 est dans la stdlib), et le WAL mode
    permet la lecture pendant qu'une écriture est en cours — utile si un
    jour plusieurs process partagent la même DB (multi-table).

Portée des tables (cf. profile_builder.py pour l'usage) :
    players               — compteurs agrégés multi-sessions (VPIP/PFR/AF...)
    player_position_stats — mêmes compteurs, ventilés par position adverse
    player_showdowns      — combos réellement vus au showdown (vérité terrain
                             pour affiner la range empirique, Palier 2)
    player_action_freq    — fréquences fold/call/raise observées par bucket
                             d'EHS (reconstruit rétroactivement à partir des
                             mains allées au showdown — cf. player_db.py),
                             c'est ce qui permet au Palier 2 de sortir du
                             mapping déterministe EHS→action des RangeBots
                             (limite documentée en phase 5, section 6.1/9.1)

Note de conception importante :
    On ne peut PAS calculer l'EHS réel d'un adversaire qui n'a pas montré
    ses cartes (il a foldé) — ce serait tricher avec une information que
    le bot n'a pas en conditions réelles. player_action_freq n'est donc
    peuplée QUE pour les mains allées au showdown, où les cartes adverses
    sont rejouées a posteriori pour reconstruire l'EHS à chaque point de
    décision. Les mains sans showdown alimentent uniquement les compteurs
    globaux (VPIP/PFR/AF), qui eux ne nécessitent aucune connaissance des
    cartes cachées. Limite assumée : le Palier 2 apprend donc plus
    lentement sur les fréquences fines que sur les tendances globales —
    cohérent avec un vrai tracker de poker (HUD), pas un raccourci du bot.

hands_seen vs preflop_opportunities (correctif session 6) :
    hands_seen compte TOUTE main où ce joueur était à table (y compris une
    main gagnée sans jamais agir — un "walk" en BB quand tout le monde
    fold avant son tour). preflop_opportunities compte uniquement les
    mains où il a RÉELLEMENT eu une décision préflop à prendre (fold ou
    entrée). VPIP/PFR doivent diviser par preflop_opportunities, jamais
    par hands_seen — sinon les walks (aucune décision, donc ni VPIP ni
    non-VPIP) diluent artificiellement le VPIP mesuré vers le bas, un peu
    comme si on comptait un absent comme "n'a pas voté" dans un scrutin.
    Bug trouvé en session 6 (diagnostic diag_vpip_trace.py) : un LAG/
    CALLING_STATION assez large finissait sous-évalué au point de passer
    pour un TAG. Les autres taux (AF, 3bet%, fold_to_cbet) étaient déjà
    corrects — ils divisent par un compteur d'opportunités dédié
    (passive_acts, hands_3bet_opp, fold_to_cbet_opp), pas par hands_seen ;
    VPIP/PFR suivent maintenant le même principe, pour rester cohérents.
"""

SCHEMA_VERSION = 1

_CREATE_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS players (
        player_id        TEXT PRIMARY KEY,
        display_name     TEXT,
        first_seen_ts     REAL,
        last_seen_ts      REAL,
        hands_seen        INTEGER NOT NULL DEFAULT 0,
        preflop_opportunities INTEGER NOT NULL DEFAULT 0,
        hands_vpip        INTEGER NOT NULL DEFAULT 0,
        hands_pfr         INTEGER NOT NULL DEFAULT 0,
        hands_3bet        INTEGER NOT NULL DEFAULT 0,
        hands_3bet_opp    INTEGER NOT NULL DEFAULT 0,
        aggressive_acts   INTEGER NOT NULL DEFAULT 0,
        passive_acts      INTEGER NOT NULL DEFAULT 0,
        fold_to_cbet_n    INTEGER NOT NULL DEFAULT 0,
        fold_to_cbet_opp  INTEGER NOT NULL DEFAULT 0,
        showdowns_seen    INTEGER NOT NULL DEFAULT 0,
        notes             TEXT DEFAULT ''
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS player_position_stats (
        player_id   TEXT NOT NULL,
        position    TEXT NOT NULL,
        hands_seen  INTEGER NOT NULL DEFAULT 0,
        hands_vpip  INTEGER NOT NULL DEFAULT 0,
        hands_pfr   INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (player_id, position),
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS player_showdowns (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id   TEXT NOT NULL,
        combo       TEXT NOT NULL,
        hand_class  TEXT NOT NULL,
        position    TEXT,
        ts          REAL,
        FOREIGN KEY (player_id) REFERENCES players(player_id)
    );
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_showdowns_player
        ON player_showdowns(player_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS player_action_freq (
        player_id   TEXT NOT NULL,
        street      TEXT NOT NULL,
        ehs_bucket  INTEGER NOT NULL,
        facing_bet  INTEGER NOT NULL,
        action      TEXT NOT NULL,
        count       INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (player_id, street, ehs_bucket, facing_bet, action)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS db_meta (
        key   TEXT PRIMARY KEY,
        value TEXT
    );
    """,
]


def _column_exists(conn, table: str, column: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


# Colonnes ajoutées après la création initiale de la table — CREATE TABLE
# IF NOT EXISTS ne modifie pas une table déjà existante, donc une DB créée
# avant l'ajout de preflop_opportunities (session 6) n'aurait pas cette
# colonne sans cette migration explicite. (table, colonne, définition SQL)
_MIGRATIONS = [
    ("players", "preflop_opportunities", "INTEGER NOT NULL DEFAULT 0"),
]


def init_schema(conn) -> None:
    """Crée les tables si absentes, applique les migrations de colonnes
    manquantes sur les DB existantes, et enregistre la version du schéma."""
    cur = conn.cursor()
    for stmt in _CREATE_STATEMENTS:
        cur.execute(stmt)

    for table, column, definition in _MIGRATIONS:
        if not _column_exists(conn, table, column):
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    cur.execute(
        "INSERT INTO db_meta (key, value) VALUES ('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
