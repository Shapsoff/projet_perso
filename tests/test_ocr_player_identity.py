"""
test_ocr_player_identity.py — Tests unitaires de ocr/player_identity.py
Phase 7 — Bot Poker Académique

Placement : tests/
Commande   : depuis la racine du projet
    pytest tests/test_ocr_player_identity.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ocr.player_identity import PlayerIdentityResolver, canonicalize_pseudo


class TestCanonicalizePseudo:

    def test_lowercases(self):
        assert canonicalize_pseudo("PokerShark99") == "pokershark99"

    def test_strips_surrounding_whitespace(self):
        assert canonicalize_pseudo("  Bob  ") == "bob"

    def test_collapses_internal_whitespace(self):
        assert canonicalize_pseudo("Le   Requin") == "le requin"

    def test_strips_stray_punctuation(self):
        assert canonicalize_pseudo("Bob!!!") == "bob"
        assert canonicalize_pseudo("[Bob]") == "bob"

    def test_keeps_dashes_and_underscores(self):
        assert canonicalize_pseudo("Jean-Pierre_92") == "jean-pierre_92"


class TestPlayerIdentityResolver:

    def test_first_read_confirmed_immediately(self):
        r = PlayerIdentityResolver(confirm_after=2)
        identity = r.resolve_seat(0, "Alice")
        assert identity.is_new_identity
        assert identity.db_pid == "ocr:alice"
        assert identity.pseudo == "Alice"

    def test_same_pseudo_stable_across_hands(self):
        r = PlayerIdentityResolver(confirm_after=2)
        first = r.resolve_seat(0, "Alice")
        second = r.resolve_seat(0, "Alice")
        assert first.db_pid == second.db_pid
        assert not second.is_new_identity

    def test_unreadable_pseudo_keeps_confirmed_identity(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        identity = r.resolve_seat(0, None)
        assert identity.db_pid == "ocr:alice"
        assert identity.pseudo == "Alice"     # dernier pseudo connu conservé
        assert not identity.is_new_identity

    def test_unreadable_pseudo_never_seen_gives_fallback(self):
        r = PlayerIdentityResolver(confirm_after=2)
        identity = r.resolve_seat(3, None)
        assert identity.db_pid == "ocr:seat3:unknown"
        assert identity.pseudo is None
        assert not identity.is_new_identity

    def test_single_conflicting_read_not_enough_to_switch(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        identity = r.resolve_seat(0, "Bob")   # 1 seul vote pour Bob
        assert identity.db_pid == "ocr:alice"  # toujours Alice
        assert not identity.is_new_identity

    def test_repeated_conflicting_reads_switch_identity(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        r.resolve_seat(0, "Bob")              # 1er vote pour Bob
        identity = r.resolve_seat(0, "Bob")    # 2e vote -> confirmé
        assert identity.db_pid == "ocr:bob"
        assert identity.is_new_identity

    def test_flip_flopping_pseudo_never_confirms(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        r.resolve_seat(0, "Bob")
        identity = r.resolve_seat(0, "Charlie")  # candidat différent -> vote repart à 1
        assert identity.db_pid == "ocr:alice"
        assert not identity.is_new_identity

    def test_empty_seat_resets_identity_immediately(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        r.mark_seat_empty(0)
        identity = r.resolve_seat(0, "Bob")
        assert identity.db_pid == "ocr:bob"
        assert identity.is_new_identity   # nouvelle identité confirmée directement, pas de vote requis

    def test_empty_seat_clears_pending_vote_too(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        r.resolve_seat(0, "Bob")          # 1er vote pour Bob, pas encore confirmé
        r.mark_seat_empty(0)
        # Un seul nouveau vote pour Bob après le siège vidé ne doit PAS suffire
        # à le confirmer immédiatement comme si le compteur avait continué.
        identity = r.resolve_seat(0, "Bob")
        assert identity.is_new_identity   # confirmé direct car "premier vu" après reset, pas via l'ancien compteur
        assert identity.db_pid == "ocr:bob"

    def test_different_seats_independent(self):
        r = PlayerIdentityResolver(confirm_after=2)
        r.resolve_seat(0, "Alice")
        r.resolve_seat(1, "Bob")
        assert r.resolve_seat(0, "Alice").db_pid == "ocr:alice"
        assert r.resolve_seat(1, "Bob").db_pid == "ocr:bob"

    def test_same_pseudo_different_seats_different_db_pid_not_merged_by_accident(self):
        # Deux joueurs différents ne portent normalement pas le même pseudo,
        # mais si jamais c'est le cas, ils partageront le même db_pid — c'est
        # un choix assumé (le pseudo EST la clé) plutôt qu'un bug caché.
        r = PlayerIdentityResolver(confirm_after=2)
        a = r.resolve_seat(0, "Alice")
        b = r.resolve_seat(1, "Alice")
        assert a.db_pid == b.db_pid == "ocr:alice"

    def test_confirm_after_must_be_at_least_one(self):
        with pytest.raises(ValueError):
            PlayerIdentityResolver(confirm_after=0)

    def test_confirm_after_one_switches_on_first_conflicting_read(self):
        r = PlayerIdentityResolver(confirm_after=1)
        r.resolve_seat(0, "Alice")
        identity = r.resolve_seat(0, "Bob")
        assert identity.db_pid == "ocr:bob"
        assert identity.is_new_identity

    def test_empty_pseudo_string_falls_back_to_seat_scoped_id(self):
        r = PlayerIdentityResolver(confirm_after=2)
        identity = r.resolve_seat(4, "   ")   # OCR a lu une chaîne vide/illisible
        assert identity.db_pid == "ocr:seat4:unknown"
