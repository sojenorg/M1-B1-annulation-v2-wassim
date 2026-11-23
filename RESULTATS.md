# Modèle d'annulation v2 — Résultats

Sur les 4 mois de test, la v2 obtient un F2 de 0,496 et la v1 rejouée 0,469. La v2 rate 218 annulations de moins sur 7 570 et produit 128 fausses alertes de plus. Les deux modèles ratent plus d'une annulation sur deux. Les calculs sont dans `notebooks/02_modelisation.ipynb`. `python train.py` reproduit le modèle et sa fiche.

<!-- toc -->
## Sommaire

- [Démarche](#démarche)
- [Données](#données)
- [Fuite de données (tâche 4)](#fuite-de-données-tâche-4)
- [Variables nouvelles](#variables-nouvelles)
- [Validation et réglage](#validation-et-réglage)
- [Choix de la métrique](#choix-de-la-métrique)
- [Verdict : v1 rejouée contre v2, sur le jeu de test](#verdict--v1-rejouée-contre-v2-sur-le-jeu-de-test)
- [Recommandation](#recommandation)
<!-- /toc -->

## Démarche

1. Données : les 78 703 réservations sont chargées par une requête SQL avec la jointure sur `hotels` (`load_data` dans `train.py`).
2. Découpage temporel : les 4 derniers mois d'arrivée (septembre à décembre 2016) forment le jeu de test. Il est lu une fois, pour le verdict.
3. Premier modèle : une forêt aléatoire sur toutes les colonnes atteint 100 % d'accuracy. Ce score a conduit à la fuite de données.
4. Référence : la recette de la v1 est rejouée sur le jeu d'entraînement et mesurée par validation croisée temporelle (`TimeSeriesSplit`, 5 folds).
5. Pipeline v2 : données brutes en entrée, 5 variables nouvelles, préparation, modèle. Trois familles sont comparées : régression logistique, forêt aléatoire, gradient boosting.
6. Réglage : 40 combinaisons d'hyperparamètres du gradient boosting et 4 de la régression logistique, tracées dans `notebooks/essais_hyperparametres.csv`.
7. Choix final : gradient boosting avec `max_depth=4`, `learning_rate=0.02`, `max_iter=300`, `class_weight="balanced"`, au seuil de décision de 0,5.

## Données

| | Période d'arrivée | Nombre de réservations | Taux d'annulation |
|---|---|---|---|
| Entraînement | 1er juillet 2015 → 31 août 2016 | 58 792 | 35,6 % |
| Test | 1er septembre → 31 décembre 2016 | 19 911 | 38,0 % |

L'exploration SQL (`notebooks/01_exploration.ipynb`) a été faite avant le découpage, sur toute la base. Les variables, les modèles et les réglages ont été choisis sur le jeu d'entraînement.

## Fuite de données (tâche 4)

**Observation.** Le premier modèle, entraîné sur toutes les colonnes, atteint 100 % d'accuracy sur 25 % du jeu d'entraînement.

**Détection.** `reservation_status_Canceled` et `reservation_status_Check-Out` portent 64 % de l'importance du modèle. Une requête SQL montre que `Canceled` et `No-Show` correspondent à `is_canceled = 1`, et `Check-Out` à 0. Les autres colonnes ont été examinées avec la même question : à quel moment l'information est-elle connue ?

**Colonnes retirées** (`LEAKY_COLUMNS` et `USELESS_COLUMNS` dans `features.py`) :

| Colonne | Moment où elle est connue | Indice |
|---|---|---|
| `reservation_status` | après le dénouement | reprend la cible |
| `reservation_status_date` | après le dénouement | 75 jours avant l'arrivée en moyenne pour une annulation, 3 jours après pour un séjour honoré |
| `assigned_room_type` | à l'arrivée | 4,3 % d'annulation si elle diffère de la chambre réservée, 41,1 % sinon |
| `booking_changes` | cumul jusqu'au dénouement | 14,1 % d'annulation avec au moins une modification, 39,2 % sans |
| `reservation_id` | identifiant | corrélation de rang de 1 avec la date d'arrivée |
| `hotel_id`, `arrival_date_year` | à la réservation | doublon de `hotel` ; année absente de l'historique pour une réservation future |

Sans `reservation_status`, l'accuracy du modèle naïf est de 91,7 %. Après retrait des autres colonnes, elle est de 91,1 %. La validation y est tirée au hasard ; la suite utilise la validation croisée temporelle.

**Colonnes gardées sous réserve.** `country`, `required_car_parking_spaces` et `total_of_special_requests` présentent des écarts forts (aucune annulation sur 3 966 réservations avec parking). Elles peuvent être complétées à l'arrivée. Elles sont gardées, car la v1 les utilise et l'API de M0-B1 les demande à la réservation. Sans elles, le F2 de validation de la v2 passe de 0,762 à 0,656.

## Variables nouvelles

Elles sont calculées dans le pipeline par `add_features` (`features.py`), à partir des données brutes. Les taux sont ceux du jeu d'entraînement.

| Variable | Calcul | Justification métier |
|---|---|---|
| `agent_id` | identifiant de l'agence en catégorie, `aucune` sans agence | les pratiques diffèrent d'une agence à l'autre : 70,4 % d'annulation pour l'agence 1, 19,4 % sans agence |
| `has_company` | 1 si une entreprise a réservé | un déplacement professionnel est moins souvent annulé : 16,9 % contre 36,7 % |
| `previous_cancel_rate` | annulations passées / réservations passées du client | l'historique est lu en proportion : 99,7 % pour les clients qui ont toujours annulé, 28,7 % sans antécédent |
| `has_children` | 1 si enfants ou bébés | un séjour en famille est moins souvent annulé : 29,1 % contre 36,1 % |
| `total_nights` | nuits de week-end + nuits de semaine | durée totale du séjour : 36,9 % de 1 à 3 nuits, 31,4 % à partir de 8 |

Elles apportent 0,037 point de F2 en validation (0,725 sans elles, 0,762 avec). L'essentiel vient de `agent_id` (0,733 sans lui). `total_nights` et `has_children` ont peu d'effet sur un modèle à arbres.

## Validation et réglage

**Validation croisée.** `TimeSeriesSplit` à 5 folds sur le jeu d'entraînement trié par date d'arrivée. Chaque fold valide sur une période postérieure à son entraînement. Avec 5 folds classiques, la v1 rejouée obtient 0,681 au lieu de 0,644.

**Familles comparées** (F2 moyen de validation) :

| Modèle | F2 validation | Écart-type | F2 entraînement | Écart |
|---|---|---|---|---|
| v1 rejouée | 0,644 | 0,194 | 0,912 | 0,268 |
| v1 rejouée + pondération des classes | 0,679 | 0,169 | 0,931 | 0,252 |
| v2 régression logistique | 0,714 | 0,074 | 0,842 | 0,128 |
| v2 forêt aléatoire | 0,688 | 0,126 | 0,912 | 0,224 |
| v2 gradient boosting, réglages de la v1 | 0,673 | 0,170 | 0,945 | 0,272 |
| v2 retenue (gradient boosting réglé) | 0,762 | 0,079 | 0,865 | 0,103 |

**Hyperparamètres.** Recherche aléatoire de 40 combinaisons sur `learning_rate`, `max_iter`, `max_depth`, `min_samples_leaf`, `l2_regularization` et `class_weight`. Retenus : `learning_rate=0.02`, `max_iter=300`, `max_depth=4`, `min_samples_leaf=20`, `l2_regularization=1`, `class_weight="balanced"`. Les trois meilleures combinaisons se tiennent en 0,005 point.

**Écart entre entraînement et validation.** Avec les réglages de la v1, le gradient boosting obtient 0,945 à l'entraînement et 0,673 en validation. La profondeur des arbres réduit le plus l'écart : 0,216 en moyenne sans limite, 0,127 à 4 niveaux. Après réglage, l'écart est de 0,103. Il ne vient pas seulement du surapprentissage : chaque fold valide sur une période plus récente. La régression logistique garde 0,128 d'écart.

**Seuil de décision.** D'après les prédictions des folds de validation, un seuil de 0,3 donnerait un F2 de 0,806 au lieu de 0,765 (rappel de 0,856 au lieu de 0,754, précision de 0,654 au lieu de 0,812). Le seuil est laissé à 0,5 pour le verdict : c'est celui de `predict`, et la v1 est comparée au même seuil.

## Choix de la métrique

- Annulation ratée : le modèle a prédit « maintenue », la chambre reste vide. L'hôtel perd le séjour.
- Annulation prédite à tort : l'hôtel relance un client qui serait venu, ou revend la chambre en surréservation. Le coût est faible pour une relance, plus élevé si un client doit être relogé.

Une annulation ratée coûte plus cher qu'une fausse alerte. La métrique principale est le F2 de la classe « annulée », où le rappel pèse deux fois plus que la précision. L'*accuracy* n'est pas retenue : elle donne le même poids aux deux erreurs.

## Verdict : v1 rejouée contre v2, sur le jeu de test

Les deux modèles sont entraînés sur le même jeu d'entraînement et évalués une fois sur les mêmes 19 911 réservations, au seuil de 0,5.

| | v1 (recette rejouée) | v2 |
|---|---|---|
| F2 (métrique principale) | 0,469 | 0,496 |
| Rappel | 0,418 | 0,447 |
| Précision | 0,924 | 0,897 |
| F1 | 0,575 | 0,596 |
| ROC AUC | 0,867 | 0,899 |
| Annulations ratées (sur 7 570) | 4 408 | 4 190 |
| Fausses alertes (sur 12 341 séjours honorés) | 262 | 390 |
| Temps d'entraînement | 1,5 s | 3,1 s |
| Taille du modèle | 0,75 Mo | 0,47 Mo |

Matrices de confusion :

| v1 rejouée | Prédit : maintenue | Prédit : annulée |
|---|---|---|
| Réel : maintenue | 12 079 | 262 |
| Réel : annulée | 4 408 | 3 162 |

| v2 | Prédit : maintenue | Prédit : annulée |
|---|---|---|
| Réel : maintenue | 11 951 | 390 |
| Réel : annulée | 4 190 | 3 380 |

Lecture :

- La v2 a un meilleur F2, un meilleur rappel et un meilleur ROC AUC. Sa précision est plus basse.
- Elle repère 218 annulations de plus et produit 128 fausses alertes de plus.
- Les scores de test sont inférieurs aux scores de validation : 0,762 et 0,496 pour la v2, 0,644 et 0,469 pour la v1.
- Annulations avec acompte non remboursable : rappel de 0,993 pour les deux modèles. Autres annulations (5 146) : 0,189 pour la v2, 0,147 pour la v1.
- La v2 prédit 30,8 % d'annulations en moyenne. Le taux réel est de 38,0 %. Les annulations ratées ont une probabilité médiane de 0,27.

## Recommandation

La v2 peut remplacer la v1 : son F2, son rappel et son ROC AUC sont meilleurs, et son fichier est plus léger. Sa précision est plus basse et son entraînement dure 3 s au lieu de 1,5 s. Elle ne répond pas entièrement au reproche du service réservations.

Réserves :

- Rappel de 0,447 sur le test : plus d'une annulation sur deux reste ratée.
- Seuil de décision : le ROC AUC est de 0,899, mais le seuil de 0,5 laisse passer des annulations. La validation croisée indique 0,3. Ce choix est à valider sur des réservations nouvelles (2017), le jeu de test ayant servi au verdict.
- Colonnes `country`, `required_car_parking_spaces` et `total_of_special_requests` : leur moment de saisie est à vérifier avec l'hôtel.
- Dérive : le score est à suivre par mois, et le modèle à ré-entraîner avec les réservations récentes. Un ré-entraînement sur les 18 mois est à envisager avant la mise en service.
- API : la v2 attend 26 colonnes brutes (liste dans `model/model_info_v2.json`) au lieu de 15, et `features.py` doit être importable là où le modèle est chargé.
