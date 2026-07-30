# Phase 6 — Player DB : notes d'intégration (v2, après relecture)

## Où placer les fichiers

```
poker_bot/
  core/
    player_db/              ← NOUVEAU package, à créer
      __init__.py
      schema.py
      player_db.py
      population_ranges.py
      profile_builder.py
      db_range_estimator.py
      db_frequency_model.py
      hand_recorder.py
    bots/
      ehs_bot.py             ← REMPLACE le fichier existant (modifs additives)
    best_response/
      best_response_engine.py ← REMPLACE le fichier existant (modifs additives)
  tests/
    test_db_range_estimator.py     ← NOUVEAU
    test_db_frequency_model.py     ← NOUVEAU
    test_ehs_bot_integration.py    ← NOUVEAU
    test_full_pipeline_integration.py ← NOUVEAU
```

`core/board_texture.py`, `core/spr.py`, `core/deck.py` ne sont PAS inclus —
ce sont tes fichiers existants, inchangés. Tu m'as fourni les deux premiers
pendant la relecture, ce qui m'a permis de tester le pipeline complet
(preflop + postflop) avec le vrai `classify_board`/`SPRInfo` plutôt qu'avec
mes stubs — voir plus bas.

## 4 bugs réels trouvés et corrigés pendant cette relecture

Tu avais raison de demander cette passe — il y avait de vrais problèmes,
tous dans les *connexions* entre `ehs_bot.py` et la Player DB (les modules
`player_db/` pris isolément étaient corrects, testés unitairement). Par
ordre de gravité :

### 1. Le gating Best-Response ignorait complètement la persistance
`BestResponseEngine._check_br_conditions()` gate sur un `hand_count` — mais
`EHSBot` lui passait `self._hand_count`, le compteur **local au match**
(remis à 0 par `full_reset()`), jamais le total cumulé en DB. Un adversaire
recroisé après une pause, pourtant déjà au Palier 2 (30+ mains DB), aurait
dû rejouer `min_hands_for_best_response` mains **dans ce match précis**
avant que le Best-Response s'active — ça videait la persistance
multi-session de l'essentiel de sa valeur.
**Fix** : `EHSBot._effective_hand_count()` = `max(compteur local, DB.hands_seen)`,
utilisé aux deux points de gating (Dimension 5 et Best-Response).

### 2. `full_reset()` sans `opponent_id` pouvait fusionner ou perdre des adversaires
`use_player_db=True` est la valeur par défaut. Or ton `validate_phase5.py`
actuel appelle `bot_v4.full_reset()` **sans argument** entre chaque match
contre un archétype différent (ligne 428). Sans correctif, l'estimateur
aurait soit gardé l'adversaire précédent, soit utilisé la chaîne littérale
`"unknown"` pour tous — fusionnant silencieusement les stats de plusieurs
adversaires distincts dans une seule entrée DB.
**Fix** : chaque `full_reset()` sans `opponent_id` génère désormais une
identité jetable et unique (`__anon_xxxxxxxxxx`), jamais réutilisée. Tant
que `hand_recorder.record_hand()` n'est pas branché, ça n'écrit d'ailleurs
rien en DB — aucune pollution, comportement strictement équivalent à la
phase 5 pour un script qui ignore `opponent_id`.
*Bonus à peu de frais : `name` (déjà dans ta boucle `for i, (name, opp) in
enumerate(opponents.items())`) est exactement l'`opponent_id` naturel si tu
veux activer la persistance dans ce script un jour — un `full_reset(name)`
suffirait.*

### 3. La position adverse pouvait fuiter d'un adversaire au suivant
Rien ne réalimentait `_villain_position` (utilisée pour le prior de
population au Palier 0) pendant le jeu. Pire : changer d'adversaire ne la
réinitialisait pas, donc le premier prior d'un nouvel adversaire pouvait
hériter à tort de la position du précédent.
**Fix** : `set_player()` réinitialise `_villain_position` à `None` ; `decide()`
la réalimente automatiquement à chaque main depuis `game_state` (effet sur
la main suivante, cohérent avec le fait que le prior de la main en cours
est déjà construit).

### 4. `in_range` était quasi toujours `True` dans le repli empirique (bug le plus subtil)
Dans `DBAwareFrequencyModel._compute_empirical()`, la branche de repli (bucket
EHS insuffisamment observé) devait reproduire `classify_combo_response()`
exactement comme en phase 5 — y compris son paramètre `in_range`, qui vaut
`False` (donc **fold garanti**, quel que soit l'EHS) pour un combo hors de
la range figée de l'archétype. Ma première version approx­imait `in_range`
par « ce combo a une masse non négligeable dans la distribution courante »
— or un prior lissé donne une masse non nulle à quasiment tous les combos
(vérifié : ~8×10⁻⁵ pour un K7o hors range TAG, bien au-dessus du seuil
`1e-6`). Résultat : `in_range` valait presque toujours `True`, et un combo
comme K7o (que le calculateur EHS stub note haut à cause du Roi, mais
qu'un TAG n'a jamais dans sa range) pouvait être classé **raise à 100%**
au lieu de **fold à 100%**.
**Fix** : `in_range` est maintenant calculé exactement comme dans
`FrequencyModel.compute()` — via `self._get_preflop_range(archetype)`
(l'ensemble réel des combos de l'archétype), pas un proxy sur la masse.
Vérifié avant/après : même scénario, ancien calcul → 100% raise, nouveau
calcul → 100% fold. Voir `tests/test_db_frequency_model.py`.

Les 4 corrections ont chacune un test de non-régression dédié (`tests/
test_ehs_bot_integration.py` pour 1-3, `tests/test_db_frequency_model.py`
pour 4).

## Testé, avec quoi (81/81 tests passent)

| Fichier | Tests | Dépendances |
|---|---|---|
| `population_ranges.py` | 8/8 | `range_definitions.py` |
| `profile_builder.py` | 16/16 | + `range_estimator.py` |
| `hand_recorder.py` | 9/9 | + `game_state.py`, `deck.py` (le vrai) |
| `test_db_range_estimator.py` | 16/16 | idem |
| `test_db_frequency_model.py` | 9/9 | + `frequency_model.py` |
| `test_ehs_bot_integration.py` | 12/12 | + `ehs_bot.py`, vrais `board_texture.py`/`spr.py` |
| `test_full_pipeline_integration.py` | 11/11 | + vrais `ev_calculator.py`, `sizing_optimizer.py`, `bluff_layer.py` |

Avec les 5 derniers fichiers reçus (`ev_calculator.py`, `sizing_optimizer.py`,
`bluff_layer.py`, `range_bot.py`, `deck.py`), j'ai enfin pu tester
`BestResponseEngine.decide()` **avec un vrai `EVCalculator`** — le seul
morceau qui manquait à la relecture précédente. Confirmé :

- `player_db=False` → `EHSBot._estimator` est un `RangeEstimator` pur et
  `_br_engine._freq_model` un `FrequencyModel` pur (aucune classe DB-aware
  impliquée) : zéro régression phase 5 vérifiée au niveau des types, pas
  seulement du comportement.
- `player_db=True` → tout le chemin `EVCalculator → SizingOptimizer →
  BluffLayer` s'exécute sans exception, avec `DBAwareFrequencyModel`
  transparent partout où `FrequencyModel` était attendu (confirmé : c'est
  bien la même instance que `EVCalculator` référence en interne).
- Le chemin empirique (Palier 2) s'active à 100% quand les buckets sont
  bien couverts, et retombe correctement sur l'archétype bucket-par-bucket
  sinon — vérifié dans le pipeline complet, pas seulement en isolation.

### Découverte (pas un bug de mes fichiers) : deux seuils de confiance désynchronisés
`best_response_engine.py` définit `_MIN_ARCHETYPE_CONFIDENCE = 0.75` avec un
commentaire daté "durci de 0.55 à 0.75 (14/07)". Mais `EHSBotConfig.
best_response_min_confidence` (dans `ehs_bot.py`) vaut encore **0.55**, et
c'est cette valeur qui gagne en pratique — `EHSBot.__init__` la passe
explicitement à `BestResponseConfig`, écrasant le défaut de 0.75. Vérifié
par calcul direct : avec ma calibration DB (`_archetype_confidence`,
plafond 0.95 atteint vers 100 mains), le seuil **effectif** de 0.55 fait
s'activer le Best-Response vers **35 mains** DB ; si tu synchronises un
jour les deux fichiers sur 0.75, ce sera plutôt vers **75 mains**. Aucune
action requise de mon côté — je te signale juste l'écart pour que le choix
soit le tien, pas un accident de synchronisation.

### Découverte (pas un bug non plus) : cas limite d'`infer_archetype()` sur les mutations très serrées
En testant avec de vraies décisions de `RangeBot` (pas des données de seed)
: un `RangeBot('TAG', mutation=0)` configuré à 5% de range est mal classé
LAG au lieu de TAG. Cause : `VillainStats.infer_archetype()` (phase 4,
`range_estimator.py`, que je ne touche pas) exige `PFR > 10%` pour la
branche TAG — or un RangeBot ne limpe jamais, donc VPIP=PFR toujours, et
toute mutation avec `range_pct ≤ 10%` échoue structurellement cette
condition. `DBAwareRangeEstimator` réutilise cette même logique à
l'identique (choix délibéré, cohérence avec la phase 4) — donc ce
comportement est rigoureusement identique en RangeEstimator classique et
en DB-aware. Pas une régression, juste une limite préexistante qui vaut
la peine d'être connue si tu as des mutations très tight dans tes 9
archétypes.

**Non testé, et ça restera hors de portée pour moi** : le vrai moteur
Monte Carlo (`poker_engine`, extension C++ compilée spécifique à ta
machine Windows). `EVCalculator._compute_equity` et `RangeBot._compute_ehs`
dégradent gracieusement vers une équité stub neutre (0.5) en son absence —
suffisant pour valider tout le câblage et la logique de décision, pas les
valeurs d'EV réalistes. `simulator.py`, lui, n'a pas de repli gracieux
(`RuntimeError` si `poker_engine` manque) — je n'ai donc pas pu exécuter
`PokerTable`/`run_match` tel quel ; à toi de valider ce dernier maillon
sur ta machine.



## Récapitulatif des fichiers/mécanismes (inchangé depuis la 1ère livraison)

Voir la section correspondante de la version précédente de ce document si
besoin — `PlayerDB` (SQLite/WAL), les 3 paliers dans `profile_builder.py`,
`DBAwareRangeEstimator`/`DBAwareFrequencyModel` en sous-classes sans
modifier une ligne des fichiers phase 4/5, `hand_recorder.record_hand()`
comme pont post-traitement. Rien de tout ça n'a changé de conception —
seules les connexions autour ont été corrigées.

## Ce qu'il reste à faire

- Le seul maillon qu'il m'est structurellement impossible de tester : le
  vrai `poker_engine` (C++ compilé pour ta machine). Tout le reste du
  pipeline (Player DB, profiling, RangeEstimator, FrequencyModel,
  EVCalculator, SizingOptimizer, BluffLayer) est maintenant validé
  bout-en-bout avec équité stub neutre. Une fois sur ta machine avec le
  vrai moteur, ce serait rassurant de relancer `tests/
  test_full_pipeline_integration.py` et vérifier que rien ne change
  structurellement (seules les valeurs d'EV changeront, pas les branches
  empruntées).
- Décider où appeler `hand_recorder.record_hand()` dans ta boucle de
  simulation réelle (`simulator.py`/`validate_phase5.py`) — juste après
  `table.run_hand()`, avec le board final et le `HandResult`.
- Si tu veux profiter de la persistance dans `validate_phase5.py` tel
  quel : changer `bot_v4.full_reset()` en `bot_v4.full_reset(opponent_id=name)`
  à la ligne 428 (une ligne, `name` existe déjà dans la boucle).

## Deux derniers correctifs (suite à ta décision sur les 2 points soulevés)

### 1. Seuil de confiance : remis à 0.55, définitivement
Décision actée : 0.55 dans `best_response_engine.py` (module constant
`_MIN_ARCHETYPE_CONFIDENCE`) ET dans `ehs_bot.py`
(`EHSBotConfig.best_response_min_confidence`) — les deux fichiers sont
maintenant alignés et le commentaire documente l'historique complet
(pourquoi 0.75 a été essayé, pourquoi ça n'a rien changé, pourquoi la
Player DB rend la question largement caduque). Aucun changement de
comportement réel puisque 0.55 était déjà la valeur effective (`ehs_bot.py`
gagnait déjà) — c'est un nettoyage de cohérence, testé (128/128).

### 2. `infer_archetype()` — deux correctifs, dans `core/range_estimator.py`
En creusant ta question "je ne comprends pas le problème", j'ai testé avec
de vraies décisions de `RangeBot` (pas des stats à la main) et trouvé que
c'était **plus large** que mon estimation initiale :

- **3/9 archétypes mal classés en déterministe** (les 3 mutations TAG,
  toutes < 7% de range — un RangeBot ne limpe jamais, donc VPIP=PFR
  toujours, et ça restait sous la barre des 10% de PFR exigée par la
  branche TAG).
- **~47% de mauvaise classification supplémentaire, mesurée sur 30 runs à
  60 mains**, pour CALLING_STATION mutation 1 (29.4% configuré) : la
  branche TAG historique ne vérifiait jamais l'AF, donc un CS passif dont
  le VPIP échantillonné tombe sous 22% par pur bruit statistique (chose
  fréquente à 60 mains) se faisait classer TAG à tort.

Deux corrections ciblées dans `VillainStats.infer_archetype()` :
1. Nouvelle branche dédiée aux ranges très serrées sans limp (`0 < v ≤
   0.10 et p ≥ 0.9·v et af ≥ 1.5` → TAG), placée avant la logique
   historique, sans la modifier.
2. Ajout de `and a >= 1.5` à la branche TAG historique — un CALLING_STATION
   reste exclu de TAG quel que soit le bruit sur son VPIP échantillonné,
   puisque son AF (passif, par construction) reste bas.

**Vérifié** : les 3 tests unitaires historiques de `test_range_estimator.py`
(TAG/LAG/CS) passent toujours à l'identique. Le taux de mauvaise
classification de CALLING_STATION mutation 1 à 60 mains est passé de 14/30
à **0/30**. 8/9 archétypes réels sont maintenant correctement classés en
déterministe (avant : 6/9).

**Non corrigé, délibérément** : LAG mutation 0 (VPIP=PFR≈20.7%) tombe
toujours dans la branche TAG historique — documenté en détail dans le
docstring de `infer_archetype()`. Le corriger proprement demande de
retoucher le seuil `v < 0.22` lui-même, qui est à seulement 0.7 point de
VPIP du cas de test historique (VPIP=0.20 attendu TAG) — un compromis plus
délicat que je n'ai pas voulu trancher unilatéralement. Dis-moi si tu veux
qu'on s'y attaque.

`tests/test_range_estimator.py` (mis à jour, section "2bis") verrouille
les deux correctifs avec des cas de non-régression dédiés, y compris le
cas dégénéré (VPIP=PFR=0%, ne doit jamais être classé TAG).

**Total : 132/132 tests passent** (85 Phase 6 + 47 `test_range_estimator.py`,
qui comptait 36 tests avant l'ajout de mes 11 nouveaux cas de régression).

## `ActionType.POST_BLIND` — correctif à la source (session 6, suite)

Un vrai run de `validate_phase6.py` sur ta machine a révélé que **tous**
les adversaires étaient classés LAG, peu importe leur vrai archétype. En
creusant (diagnostic dédié, stats brutes en DB) : VPIP=PFR=100% pour tout
le monde. Cause trouvée dans ton propre `simulator.py` — un TODO déjà
noté (`# ❌ Créer une ActionType.POST_BLIND plutot que RAISE quand c'est
les blindes`) : les mises forcées (SB/BB) étaient loggées comme
`ActionType.RAISE`, indiscernables d'une vraie relance volontaire.

**Corrigé à la source, pas contourné**, comme demandé :

- `game_state.py` : ajout de `ActionType.POST_BLIND = "post_blind"`.
- `simulator.py::_post_blindes()` : utilise ce nouveau type, TODO supprimé.
- `hand_recorder.py` : **simplifié** — l'heuristique de détection par
  montant (`small_blind`/`big_blind` en paramètres) devient inutile et a
  été retirée. `POST_BLIND` n'étant dans aucun des tuples
  `_VPIP_TYPES`/`_AGGRESSIVE_TYPES`, les blindes sont exclues nativement,
  sans code de filtrage dédié.
- `validate_phase6.py` : mis à jour en conséquence (plus besoin de passer
  les montants de blindes à `record_hand()`).

**Effet de bord découvert et vérifié empiriquement** — ce même bug
touchait aussi deux autres fichiers, sans qu'aucune modification
supplémentaire n'ait été nécessaire (juste la conséquence de corriger à
la source) :

- `range_estimator.py` (phase 4, non modifié) : `VillainStats._stats`
  comptait aussi les blindes comme VPIP/PFR — vérifié avant/après, un
  simple poste-puis-fold passe de `hands_vpip=1` à `hands_vpip=0`.
- `action_history.py` (Dimension 2, non modifié) : **plus grave** — une
  simple **ouverture** préflop était classée dans le bucket
  `SQUEEZE_4BET` (le plus extrême des 12, "4bet ou squeeze premium"), à
  cause du comptage global des "raises" qui incluait les 2 blindes.
  Vérifié avant/après : `OPEN_CALL_PASSIVE` correctement après le fix.
  Ce bug touchait potentiellement **chaque main jouée en heads-up depuis
  la phase 2/3**, pas seulement la Phase 6 — à garder en tête si tu
  observais des comportements Dimension 2 étranges dans les phases
  précédentes.

Tests mis à jour : `hand_recorder.py` (13/13, dont un test dédié qui
vérifie qu'un poste-puis-fold ne compte plus comme VPIP, et qu'un open
après les blindes n'est jamais compté comme 3bet).

**Suite à ta demande de corriger à la racine plutôt que survoler** :
`action_history.py` et `range_estimator.py` sont maintenant modifiés
explicitement (ils fonctionnaient déjà correctement par coïncidence —
`'post_blind'` ne matchait aucune chaîne reconnue — mais c'était fragile
et implicite, pas un vrai comportement voulu et documenté) :

- `range_estimator.py` : `_update_preflop()` et `_update_stats_from_action()`
  sortent maintenant explicitement dès qu'ils reçoivent `'post_blind'`
  (avant de faire quoi que ce soit), avec un commentaire qui référence le
  bug trouvé. Effet de bord positif : plus efficace aussi (évite
  d'itérer les 1326 combos pour rien).
- `action_history.py` : `extract_signals()` exclut maintenant
  explicitement `'post_blind'` en tête de boucle, avant tout comptage.
  **3 nouveaux tests permanents ajoutés** (16/16 au total, contre 13
  avant), qui reproduisent exactement le bug (open → à tort classé
  4bet) avec des blindes réalistes en préfixe, en utilisant le tag
  générique `'raise'` — pas `'3bet'`/`'4bet'` explicites, que les vrais
  bots (RangeBot/EHSBot) n'utilisent JAMAIS. Ça confirme au passage que
  le chemin par comptage (`n_before`) que j'ai corrigé est le SEUL
  chemin réellement emprunté en jeu — les branches `elif action=='3bet'`
  étaient déjà quasiment du code mort pour ce projet.
- Nettoyage en prime : une variable `raises_before` calculée mais jamais
  utilisée (code mort, probablement un reliquat d'un refactor antérieur)
  a été supprimée dans `action_history.py`.
- `test_range_estimator.py` : nouveau `test_post_blind_handling()`
  (3 tests) qui vérifie qu'un `post_blind` seul ne modifie ni la
  distribution ni les compteurs VPIP/PFR, et qu'un `post_blind + fold`
  est rigoureusement identique à un `fold` seul.

**Total : 151/151 tests passent** (85 Phase 6 + 16 `action_history.py`
+ 50 `test_range_estimator.py`).

## `tests/validate_phase6.py` — nouveau, ne touche pas à `validate_phase5.py`

Utilise ton vrai `simulator.py`/`PokerTable` (donc nécessite ton `poker_engine`
compilé, indisponible ici — je n'ai pu tester que les morceaux isolables :
l'adaptateur `_make_bot_fn`+capture du board, et `run_match_with_recording()`
avec une fausse `PokerTable`, tous deux vérifiés fonctionnels). À exécuter
chez toi :

```
python tests/validate_phase6.py --quick             # 200 mains/match, tous les RangeBots
python tests/validate_phase6.py --opponent TAG_0    # un seul adversaire
python tests/validate_phase6.py --db_path data/ma_db.sqlite3  # DB nommée, inspectable après coup
```

Ce que ça mesure, avec de VRAIES mains jouées (pas de seed synthétique) :
1. **Convergence** : après N mains contre chaque RangeBot, le profil DB
   retrouve-t-il le bon archétype (palier ≥ 1) ? Rapport final : X/9 corrects.
2. **Persistance multi-session** : ferme la DB, en rouvre une nouvelle
   connexion (simule un nouveau process), vérifie que le profil a survécu
   à l'identique.

Si `poker_engine` est disponible (ton cas), les showdowns sont utilisés pour
peupler les fréquences empiriques par bucket d'EHS (Palier 2) via le vrai
calculateur — sinon (repli automatique) seules les stats VPIP/PFR/AF sont
enregistrées, les fréquences empiriques restent vides et le Palier 2 se
limite au mélange archétype/showdown.

## Dilution du VPIP par les mains "walk" (session 6, suite)

Après le correctif POST_BLIND, `validate_phase6.py --quick` a montré un
nouveau biais : tout le monde tendait vers TAG (5/9 corrects, mais 4 des 5
erreurs classaient à tort en TAG). Diagnostiqué avec `tests/
diag_vpip_trace.py` (nouveau, gardé dans le repo) : sur un adversaire
factice qui fold toujours, `combo_in_range()=True` correspondait
EXACTEMENT à `hands_vpip` en DB (aucun bug d'enregistrement) — mais le
VPIP mesuré (`hands_vpip / hands_seen`) sortait systématiquement à la
MOITIÉ du bon ratio.

**Cause** : `hands_seen` compte toute main où le joueur était à table, y
compris les mains "walk" (il gagne sans jamais avoir eu de décision
préflop à prendre — ex: BB quand tout le monde fold avant son tour).
Diviser le VPIP par `hands_seen` dilue artificiellement le résultat vers
le bas — une main sans décision n'est ni un VPIP ni un non-VPIP, elle
devrait juste être exclue du calcul, pas comptée comme "n'a pas VPIP".
Ce n'est pas spécifique au diagnostic : ça affecte aussi les vraies
parties EHSBot vs RangeBot (en plus léger, puisqu'EHSBot ne fold pas
systématiquement en SB) — cohérent avec le fait que seuls les archétypes
proches du seuil de 22% (LAG_0/1, CALLING_STATION_0/1) basculaient à
tort, pas les plus larges (LAG_2, CALLING_STATION_2).

Point notable en creusant : AF, 3bet% et fold-to-cbet utilisaient déjà
tous un dénominateur "opportunités" correct (`passive_acts`,
`hands_3bet_opp`, `fold_to_cbet_opp`) — seuls VPIP/PFR divisaient par
`hands_seen`, une incohérence avec le reste du fichier.

**Correctif** (`schema.py`, `player_db.py`, `profile_builder.py`) :
- Nouvelle colonne `preflop_opportunities`, incrémentée uniquement dans
  `record_preflop_action()` — qui n'est déjà appelée par `hand_recorder.py`
  que lorsqu'une vraie décision a eu lieu (`if not mine: return` déjà en
  place), donc aucun risque de compter une main "walk" par erreur.
- `PlayerRow.vpip`/`.pfr` divisent maintenant par `preflop_opportunities`.
- **Le vrai correctif du bug observé** : `profile_builder.py` reconstruit
  un `VillainStats` (pour réutiliser `infer_archetype()` de
  `range_estimator.py`) à partir des champs bruts de `PlayerRow`, PAS de
  ses propriétés `.vpip`/`.pfr` — il passait donc `hands_seen=row.hands_seen`
  directement, contournant même la correction ci-dessus. Corrigé pour
  passer `hands_seen=row.preflop_opportunities`.
- Migration automatique (`ALTER TABLE`) pour toute DB déjà créée avant ce
  correctif — pas besoin de supprimer tes fichiers `.sqlite3` existants.

**Délibérément laissé inchangé** : le palier (Tier 0/1/2) et
`PlayerProfile.hands_seen` (utilisé par `EHSBot._effective_hand_count()`
pour le gating Best-Response) continuent d'utiliser `hands_seen` brut, pas
`preflop_opportunities`. Les toucher aurait un effet de bord bien plus
large (plusieurs tests existants, le gating Best-Response) pour un gain
incertain — les mains "walk" dépendent de NOTRE taux de fold en SB, pas
de l'adversaire tracké, donc devraient rester rares en jeu réel dynamique
(contrairement à mon diagnostic, qui utilise un adversaire qui fold
100% du temps pour forcer le cas extrême). Nouveau test de régression
dédié (`profile_builder.py`, "dilution session 6") qui vérifie qu'un LAG
à VPIP réel 25% reste correctement classé LAG même avec 50% de mains
walk — pire que ce à quoi il faut s'attendre en jeu réel.

**Total : 155/155 tests passent.**


