"""
player_db.py — Couche de persistance de la Player DB
Phase 6 — Bot Poker Académique

Responsabilité unique : lire/écrire les statistiques joueur en SQLite.
Ce module ne contient AUCUNE logique de profiling (ranges, archétypes,
paliers de cold start) — c'est le rôle de profile_builder.py. Séparation
volontaire, dans la continuité de l'architecture des phases précédentes
(range_definitions.py = données, range_estimator.py = logique).

Identité joueur :
    player_id: str — neutre par construction. En simulation, c'est le nom
    du bot ("TAG_0", "CALLING_STATION_2", un hash de seed pour le
    self-play...). En conditions réelles (phase 7, OCR), ce sera
    l'identifiant remonté par la reconnaissance d'écran (pseudo table).
    Aucune méthode de cette classe ne connaît ou ne suppose la notion de
    "RangeBot" — cette couche est donc structurellement identique qu'on
    soit en test ou en conditions réelles.

Concurrence :
    WAL mode activé à l'ouverture — permet à un lecteur (ex: outil
    d'analyse externe) de consulter la DB pendant qu'une partie est en
    cours d'écriture, sans bloquer.

Usage typique :
    db = PlayerDB("data/player_db.sqlite3")
    db.get_or_create_player("TAG_0")
    db.record_preflop_action("TAG_0", position="BTN", vpip=True, pfr=True)
    db.record_postflop_aggression("TAG_0", aggressive=True)
    row = db.get_player_row("TAG_0")
    db.close()
"""

from __future__ import annotations

import sqlite3
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.bots.range_definitions import RANK_VAL, SUITS
from .schema import init_schema

# Nombre de buckets d'EHS utilisés pour player_action_freq (déciles).
N_EHS_BUCKETS = 10

# Ordre canonique des combos — DOIT rester identique à celui utilisé pour
# construire range_estimator.ALL_COMBOS (cartes ordonnées par rang puis
# couleur). Sans cette canonicalisation à l'écriture, un combo stocké
# "AhKd" ne matcherait jamais la clé "KdAh" utilisée dans ALL_COMBOS côté
# lecture (profile_builder.get_preflop_prior) — bug silencieux (masse
# perdue), détecté par les tests de profile_builder.py.
_SUIT_IDX = {s: i for i, s in enumerate(SUITS)}


def canonical_combo(card1: str, card2: str) -> str:
    """Ordonne deux cartes selon la même convention que ALL_COMBOS."""
    def _key(c: str):
        return (RANK_VAL[c[0]], _SUIT_IDX[c[1]])
    if _key(card1) <= _key(card2):
        return card1 + card2
    return card2 + card1


def ehs_to_bucket(ehs: float, n_buckets: int = N_EHS_BUCKETS) -> int:
    """Convertit un EHS ∈ [0,1] en index de bucket ∈ [0, n_buckets-1]."""
    ehs = max(0.0, min(1.0, ehs))
    bucket = int(ehs * n_buckets)
    return min(bucket, n_buckets - 1)


@dataclass
class PlayerRow:
    """Vue typée d'une ligne de la table players."""
    player_id:        str
    hands_seen:        int = 0
    preflop_opportunities: int = 0
    hands_vpip:        int = 0
    hands_pfr:         int = 0
    hands_3bet:        int = 0
    hands_3bet_opp:    int = 0
    aggressive_acts:   int = 0
    passive_acts:      int = 0
    fold_to_cbet_n:    int = 0
    fold_to_cbet_opp:  int = 0
    showdowns_seen:    int = 0

    @property
    def vpip(self) -> float:
        """
        VPIP = hands_vpip / preflop_opportunities — PAS hands_seen.

        hands_seen compte aussi les mains "walk" (le joueur gagne sans
        jamais avoir eu à décider, ex: BB quand tout le monde fold avant
        son tour) — les diviser par hands_seen diluerait artificiellement
        le VPIP vers le bas (bug trouvé en session 6, cf. schema.py pour
        le détail). preflop_opportunities ne compte que les mains où une
        vraie décision a eu lieu, comme aggressive_acts/passive_acts pour
        l'AF ou hands_3bet_opp pour le 3bet%.
        """
        return (self.hands_vpip / self.preflop_opportunities
                if self.preflop_opportunities > 0 else 0.5)

    @property
    def pfr(self) -> float:
        return (self.hands_pfr / self.preflop_opportunities
                if self.preflop_opportunities > 0 else 0.3)

    @property
    def three_bet_pct(self) -> float:
        return (self.hands_3bet / self.hands_3bet_opp
                if self.hands_3bet_opp > 0 else 0.0)

    @property
    def af(self) -> float:
        return (self.aggressive_acts / self.passive_acts
                if self.passive_acts > 0 else 3.0)

    @property
    def fold_to_cbet(self) -> float:
        return (self.fold_to_cbet_n / self.fold_to_cbet_opp
                if self.fold_to_cbet_opp > 0 else 0.5)


class PlayerDB:
    """Persistance SQLite des statistiques joueur, multi-session."""

    def __init__(self, db_path: str = "data/player_db.sqlite3"):
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        # Sans ceci, un writer qui ne peut pas acquérir immédiatement le
        # verrou lève "database is locked" au lieu d'attendre — un risque
        # réel si plusieurs process/threads de simulation écrivent en
        # parallèle sur la même DB (cf. multi_test_sim_*.py). 5s laisse le
        # temps à l'autre writer de finir sans bloquer indéfiniment.
        self._conn.execute("PRAGMA busy_timeout=5000;")
        init_schema(self._conn)

    # =========================================================================
    # Gestion du joueur
    # =========================================================================

    def get_or_create_player(self, player_id: str, display_name: str = "") -> None:
        now = time.time()
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO players (player_id, display_name, first_seen_ts, last_seen_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(player_id) DO UPDATE SET last_seen_ts=excluded.last_seen_ts
            """,
            (player_id, display_name, now, now),
        )
        self._conn.commit()

    def get_player_row(self, player_id: str) -> Optional[PlayerRow]:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return PlayerRow(
            player_id=row["player_id"],
            hands_seen=row["hands_seen"],
            preflop_opportunities=row["preflop_opportunities"],
            hands_vpip=row["hands_vpip"],
            hands_pfr=row["hands_pfr"],
            hands_3bet=row["hands_3bet"],
            hands_3bet_opp=row["hands_3bet_opp"],
            aggressive_acts=row["aggressive_acts"],
            passive_acts=row["passive_acts"],
            fold_to_cbet_n=row["fold_to_cbet_n"],
            fold_to_cbet_opp=row["fold_to_cbet_opp"],
            showdowns_seen=row["showdowns_seen"],
        )

    def list_players(self) -> List[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT player_id FROM players ORDER BY hands_seen DESC")
        return [r["player_id"] for r in cur.fetchall()]

    # =========================================================================
    # Enregistrement — compteurs globaux (ne nécessitent pas de connaître
    # les cartes cachées de l'adversaire — disponibles en conditions réelles)
    # =========================================================================

    def new_hand_observed(self, player_id: str) -> None:
        """Incrémente hands_seen. Appelé une fois par main terminée."""
        self.get_or_create_player(player_id)
        self._conn.execute(
            "UPDATE players SET hands_seen = hands_seen + 1, last_seen_ts = ? "
            "WHERE player_id = ?",
            (time.time(), player_id),
        )
        self._conn.commit()

    def record_preflop_action(
        self,
        player_id:  str,
        position:   Optional[str],
        vpip:       bool,
        pfr:        bool,
        is_3bet:               bool = False,
        faced_3bet_opportunity: bool = False,
    ) -> None:
        """
        Enregistre une action preflop pour les compteurs VPIP/PFR/3bet.
        À appeler une fois par main pour l'action preflop la plus
        significative du joueur (son action d'entrée dans le pot) —
        c'est-à-dire uniquement quand ce joueur a RÉELLEMENT eu une
        décision préflop à prendre. hand_recorder.py garantit déjà cet
        appel conditionnel (jamais pour une main "walk" sans décision) ;
        preflop_opportunities s'incrémente donc ici en toute sécurité à
        chaque appel, sans jamais compter une main où aucune décision
        n'a eu lieu — cf. schema.py pour le détail du bug évité.
        """
        self.get_or_create_player(player_id)
        cur = self._conn.cursor()
        cur.execute(
            """
            UPDATE players SET
                preflop_opportunities = preflop_opportunities + 1,
                hands_vpip     = hands_vpip     + ?,
                hands_pfr      = hands_pfr      + ?,
                hands_3bet     = hands_3bet     + ?,
                hands_3bet_opp = hands_3bet_opp + ?
            WHERE player_id = ?
            """,
            (int(vpip), int(pfr), int(is_3bet), int(faced_3bet_opportunity),
             player_id),
        )
        if position:
            cur.execute(
                """
                INSERT INTO player_position_stats
                    (player_id, position, hands_seen, hands_vpip, hands_pfr)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(player_id, position) DO UPDATE SET
                    hands_seen = hands_seen + 1,
                    hands_vpip = hands_vpip + excluded.hands_vpip,
                    hands_pfr  = hands_pfr  + excluded.hands_pfr
                """,
                (player_id, position.upper(), int(vpip), int(pfr)),
            )
        self._conn.commit()

    def record_postflop_aggression(self, player_id: str, aggressive: bool) -> None:
        """Un acte postflop observé : bet/raise (True) ou call/check (False)."""
        self.get_or_create_player(player_id)
        col = "aggressive_acts" if aggressive else "passive_acts"
        self._conn.execute(
            f"UPDATE players SET {col} = {col} + 1 WHERE player_id = ?",
            (player_id,),
        )
        self._conn.commit()

    def record_fold_to_cbet(self, player_id: str, folded: bool) -> None:
        """À appeler uniquement quand le joueur FAIT FACE à une c-bet."""
        self.get_or_create_player(player_id)
        self._conn.execute(
            "UPDATE players SET fold_to_cbet_opp = fold_to_cbet_opp + 1, "
            "fold_to_cbet_n = fold_to_cbet_n + ? WHERE player_id = ?",
            (int(folded), player_id),
        )
        self._conn.commit()

    # =========================================================================
    # Enregistrement — données de showdown (nécessitent les cartes révélées)
    # =========================================================================

    def record_showdown_combo(
        self,
        player_id:  str,
        combo:      str,
        hand_class: str,
        position:   Optional[str] = None,
    ) -> None:
        """
        Enregistre une main réellement montrée au showdown.

        `combo` peut être fourni dans n'importe quel ordre de cartes
        ("AhKd" ou "KdAh") — il est canonicalisé avant stockage pour rester
        cohérent avec range_estimator.ALL_COMBOS (cf. canonical_combo()).
        """
        combo = canonical_combo(combo[:2], combo[2:])
        self.get_or_create_player(player_id)
        self._conn.execute(
            """
            INSERT INTO player_showdowns (player_id, combo, hand_class, position, ts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (player_id, combo, hand_class, position, time.time()),
        )
        self._conn.execute(
            "UPDATE players SET showdowns_seen = showdowns_seen + 1 "
            "WHERE player_id = ?",
            (player_id,),
        )
        self._conn.commit()

    def record_ehs_bucket_action(
        self,
        player_id:  str,
        street:     str,
        ehs:        float,
        facing_bet: bool,
        action:     str,
    ) -> None:
        """
        Enregistre une action à un point de décision dont l'EHS réel est
        connu (reconstruit a posteriori depuis un showdown — cf. schema.py).
        """
        bucket = ehs_to_bucket(ehs)
        self.get_or_create_player(player_id)
        self._conn.execute(
            """
            INSERT INTO player_action_freq
                (player_id, street, ehs_bucket, facing_bet, action, count)
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(player_id, street, ehs_bucket, facing_bet, action)
            DO UPDATE SET count = count + 1
            """,
            (player_id, street, bucket, int(facing_bet), action),
        )
        self._conn.commit()

    # =========================================================================
    # Lecture — agrégats pour le profile_builder
    # =========================================================================

    def get_position_stats(self, player_id: str) -> Dict[str, dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM player_position_stats WHERE player_id = ?",
            (player_id,),
        )
        return {
            r["position"]: {
                "hands_seen": r["hands_seen"],
                "hands_vpip": r["hands_vpip"],
                "hands_pfr":  r["hands_pfr"],
            }
            for r in cur.fetchall()
        }

    def get_showdown_combos(self, player_id: str) -> List[Tuple[str, str, Optional[str]]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT combo, hand_class, position FROM player_showdowns "
            "WHERE player_id = ?",
            (player_id,),
        )
        return [(r["combo"], r["hand_class"], r["position"]) for r in cur.fetchall()]

    def get_action_freq_table(
        self, player_id: str
    ) -> Dict[Tuple[str, int, int], Dict[str, int]]:
        """
        Retourne {(street, ehs_bucket, facing_bet): {action: count}}.
        Structure directement consommable par profile_builder pour bâtir
        des fréquences fold/call/raise lissées par bucket.
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT street, ehs_bucket, facing_bet, action, count "
            "FROM player_action_freq WHERE player_id = ?",
            (player_id,),
        )
        table: Dict[Tuple[str, int, int], Dict[str, int]] = {}
        for r in cur.fetchall():
            key = (r["street"], r["ehs_bucket"], r["facing_bet"])
            table.setdefault(key, {})[r["action"]] = r["count"]
        return table

    # =========================================================================
    # Cycle de vie
    # =========================================================================

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PlayerDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"PlayerDB(path={self.db_path!r})"
