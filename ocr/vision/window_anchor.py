"""
window_anchor.py — Localisation de la fenêtre cible, indépendante de sa
position (et, combiné à region_config.py, de sa taille) à l'écran.
Phase 7 — Bot Poker Académique

Principe : au lieu de capturer des coordonnées ABSOLUES à l'écran (qui
cassent dès que la fenêtre du client est déplacée), on récupère à chaque
frame la zone cliente RÉELLE de la fenêtre (son coin haut-gauche + sa
largeur/hauteur actuels), et toutes les régions calibrées (cf.
region_config.py) sont résolues relativement à cette zone. Déplacer la
fenêtre ne casse donc plus rien ; redimensionner ne casse plus rien non
plus SI region_config.py est utilisé avec des coordonnées fractionnelles
(0..1) plutôt qu'en pixels absolus — combiner les deux modules donne un
système robuste à la fois au déplacement ET au redimensionnement, tant
que le client redessine son contenu proportionnellement (vrai pour la
quasi-totalité des clients modernes, souvent basés sur du rendu
web/Electron).

"Zone cliente" et pas juste le rectangle de la fenêtre : le rectangle de
fenêtre inclut la barre de titre et les bordures, dont la hauteur ne fait
pas partie du plateau de jeu — l'exclure rend la calibration plus précise
et plus stable d'un thème Windows à l'autre.

Fenêtres "pop-out" (une table de poker ouverte dans sa propre fenêtre,
recréée à chaque fois qu'on ouvre/ferme une table plutôt que réutilisée) :
Win32WindowLocator ne mémorise JAMAIS un handle de fenêtre (hwnd) d'un
appel à l'autre — il refait une recherche complète (EnumWindows) à
CHAQUE appel de find_client_bbox(). Une nouvelle instance de fenêtre,
même avec un hwnd Windows différent de la précédente, est donc retrouvée
normalement au prochain appel du moment que son titre correspond
toujours. Le vrai risque n'est pas là : c'est quand PLUSIEURS fenêtres
correspondent en même temps (plusieurs tables ouvertes simultanément,
multi-tabling) — la recherche ne devine alors jamais laquelle prendre,
elle lève AmbiguousWindowError plutôt que de choisir arbitrairement (lire
silencieusement les données de la mauvaise table serait le pire des
bugs possibles pour un bot qui joue de l'argent réel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple


class AmbiguousWindowError(Exception):
    """
    Plusieurs fenêtres visibles correspondent au même title_substring —
    typiquement plusieurs tables ouvertes simultanément (multi-tabling)
    avec des titres trop similaires pour les distinguer. Levée plutôt que
    de choisir arbitrairement, pour ne jamais risquer de lire la mauvaise
    table en silence.
    """


@dataclass(frozen=True)
class BoundingBox:
    """Rectangle en coordonnées écran absolues, en pixels."""
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"BoundingBox de taille invalide : {self.width}x{self.height}")

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_mss_dict(self) -> dict:
        """Format attendu par mss.mss().grab()."""
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class WindowLocator(Protocol):
    """
    Abstraction sur la source des coordonnées de fenêtre. Permet de
    tester tout le reste du pipeline (region_config, frame_capture) sans
    dépendre de pywin32 (Windows uniquement, absent de certains
    environnements de dev/CI) — cf. StaticWindowLocator plus bas pour les
    tests, et Win32WindowLocator pour l'usage réel.
    """
    def find_client_bbox(self) -> Optional[BoundingBox]: ...


class StaticWindowLocator:
    """
    WindowLocator de test / de développement : renvoie toujours la bbox
    fournie explicitement (modifiable via set_bbox, pour simuler un
    déplacement ou un redimensionnement dans un test). Utile aussi pour
    développer le reste du pipeline sans avoir de vraie fenêtre de client
    de poker ouverte.
    """

    def __init__(self, bbox: Optional[BoundingBox]) -> None:
        self._bbox = bbox

    def find_client_bbox(self) -> Optional[BoundingBox]:
        return self._bbox

    def set_bbox(self, bbox: Optional[BoundingBox]) -> None:
        self._bbox = bbox


class Win32WindowLocator:
    """
    Implémentation réelle, Windows uniquement (`pip install pywin32`).
    Recherche une fenêtre visible dont le titre contient
    `title_substring` (insensible à la casse), et renvoie sa zone
    cliente en coordonnées écran.

    L'import de win32gui est fait DANS les méthodes plutôt qu'en tête de
    module pour que ce fichier reste importable sur un environnement sans
    pywin32 (Linux, macOS, CI) — seul l'appel réel échoue alors, pas
    l'import du module ni celui du package ocr.vision dans son ensemble.

    Si plusieurs tables sont ouvertes simultanément avec des titres qui se
    ressemblent, title_substring seul ne suffit pas à les distinguer —
    vérifie le titre RÉEL de tes fenêtres de table pop-out (imprime
    win32gui.GetWindowText pour chaque fenêtre visible) avant de choisir
    ta sous-chaîne : si le client inclut un identifiant de table dans le
    titre (nom de table, stakes...), utilise-le pour cibler précisément
    UNE table.

    class_name (optionnel) filtre EN PLUS sur la classe EXACTE de la
    fenêtre (cf. list_windows_v2.list_visible_windows_with_classes() pour
    la repérer) — nécessaire dès que le client ouvre plusieurs fenêtres
    qui partagent un titre proche mais pas la même classe (typiquement le
    lobby ET une table contenant toutes deux le nom du client dans leur
    titre) : title_substring seul les confondrait et lèverait
    AmbiguousWindowError, alors que class_name les distingue proprement.
    Si ça ne suffit toujours pas (plusieurs tables identiques ouvertes en
    multi-tabling), il faudra une stratégie supplémentaire (PID du
    processus, ou choix manuel du hwnd) — à ajouter si besoin.

    include_titlebar (optionnel, défaut False) : si True, find_client_bbox()
    retourne malgré son nom la bbox de la fenêtre ENTIÈRE (GetWindowRect —
    bordures + barre de titre incluses) plutôt que la seule zone cliente
    (GetClientRect). Le nom de la méthode n'a pas changé pour ne rien
    casser dans le reste du pipeline (frame_capture.py et tout ce qui en
    dépend appellent tous find_client_bbox() sans savoir ce qu'il y a
    dedans). À utiliser UNIQUEMENT si l'image de référence utilisée pour
    calibrer (cf. capture_reference.py / calibration_tool.py) a elle
    aussi été capturée fenêtre entière — sinon les deux ne correspondent
    plus et regions.json devient faux (décalage + mauvaise échelle).

    Compromis à connaître : le choix par défaut (zone cliente seule) est
    délibéré (cf. docstring de module) car la hauteur du bandeau de titre
    n'est PAS proportionnelle à la taille de la zone de jeu — elle varie
    selon le thème Windows / le DPI. Inclure le bandeau rend donc la
    calibration un peu moins portable d'une machine à l'autre. Si tu
    changes de machine ou de thème plus tard, une petite dérive sur les
    régions du haut de la table (souvent hero_card_*, seat du haut) est
    plus probable qu'avec le mode par défaut.
    """

    def __init__(
        self,
        title_substring: str,
        class_name: Optional[str] = None,
        include_titlebar: bool = False,
    ) -> None:
        if not title_substring:
            raise ValueError("title_substring ne peut pas être vide")
        self.title_substring = title_substring
        self.class_name = class_name
        self.include_titlebar = include_titlebar
        # Rempli à chaque appel de find_client_bbox() — lu par
        # capture_table_frame() pour transmettre le hwnd à un grabber qui
        # en a besoin (ex: PrintWindowFrameGrabber), sans refaire une
        # recherche EnumWindows séparée pour l'obtenir.
        self.last_hwnd: Optional[int] = None

    def find_client_bbox(self) -> Optional[BoundingBox]:
        import win32gui

        hwnd = self._find_hwnd()
        self.last_hwnd = hwnd
        if hwnd is None:
            return None

        if self.include_titlebar:
            # Fenêtre ENTIÈRE : GetWindowRect renvoie déjà des coordonnées
            # écran absolues, pas besoin de ClientToScreen ici.
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width, height = right - left, bottom - top
            if width <= 0 or height <= 0:
                return None
            return BoundingBox(left, top, width, height)

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            return None  # fenêtre minimisée ou masquée

        screen_left, screen_top = win32gui.ClientToScreen(hwnd, (left, top))
        return BoundingBox(screen_left, screen_top, width, height)

    def _find_hwnd(self) -> Optional[int]:
        import win32gui

        candidates: List[Tuple[int, str, str]] = []

        def _callback(hwnd: int, _extra: object) -> None:
            if win32gui.IsWindowVisible(hwnd):
                candidates.append(
                    (hwnd, win32gui.GetWindowText(hwnd), win32gui.GetClassName(hwnd))
                )

        win32gui.EnumWindows(_callback, None)
        return select_window(candidates, self.title_substring, self.class_name)


def select_window(
    candidates: List[Tuple[int, str, str]],
    title_substring: str,
    class_name: Optional[str] = None,
) -> Optional[int]:
    """
    Choisit un hwnd parmi une liste déjà énumérée de (hwnd, titre, classe).

    Séparée de l'énumération réelle (EnumWindows, Windows uniquement)
    pour rester une fonction PURE, testable sans pywin32 ni fenêtre
    réelle — c'est elle qui porte toute la logique de désambiguïsation,
    Win32WindowLocator._find_hwnd ne fait que lui fournir les candidats.

    class_name, si fourni, filtre EN PLUS sur la classe EXACTE (pas de
    sous-chaîne, contrairement au titre) — cf. docstring de
    Win32WindowLocator pour le cas d'usage (lobby vs table de même
    client).

    NOTE DE MIGRATION : le format des tuples de `candidates` est passé de
    (hwnd, titre) à (hwnd, titre, classe) — si tu as des tests qui
    appellent cette fonction directement avec des tuples à 2 éléments,
    il faut leur ajouter un 3e champ (classe, ou "" si non pertinent).

    Raises:
        AmbiguousWindowError si plus d'une fenêtre correspond (après
        filtrage par classe, le cas échéant).
    """
    needle = title_substring.lower()
    matches = [
        hwnd
        for hwnd, title, wcls in candidates
        if needle in title.lower() and (class_name is None or wcls == class_name)
    ]

    if len(matches) > 1:
        where = repr(title_substring)
        if class_name is not None:
            where += f" (classe {class_name!r})"
        raise AmbiguousWindowError(
            f"{len(matches)} fenêtres visibles correspondent à {where} — "
            "impossible de choisir automatiquement laquelle est la bonne "
            "table. Précise le titre, ajoute/corrige class_name (cf. "
            "list_windows_v2.py pour repérer la classe), ou utilise une "
            "autre stratégie de sélection (PID) si plusieurs tables "
            "identiques sont ouvertes en même temps."
        )
    return matches[0] if matches else None
