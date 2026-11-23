# Escale Atlantique — Ré-entraînement du modèle d'annulation (v2)

> Brief **M1-B1** (FastIA), réalisé à partir du squelette `twentz29/Dev-IA-parcours-m1-b1`.

<!-- toc -->
## Sommaire

- [Le contexte](#le-contexte)
- [Structure](#structure)
- [Démarrage](#démarrage)
- [Ré-entraîner le modèle](#ré-entraîner-le-modèle)
- [La base](#la-base)
- [Données](#données)
<!-- /toc -->

## Le contexte

FastIA reprend le modèle de prédiction d'annulation d'Escale Atlantique, exposé en M0-B1. L'objectif : le ré-entraîner proprement à partir de l'historique des réservations, et démontrer qu'il fait mieux que la v1.

## Structure

```
docker-compose.yml        # base PostgreSQL locale
.env.example              # identifiants de la base locale, à copier en .env
db/init/                  # schéma + historique chargés au 1er démarrage de la base
  01_schema.sql
  02_load.sql
  data/reservations.csv   # 78 703 réservations, juillet 2015 → décembre 2016
notebooks/
  01_exploration.ipynb    # tâches 1 à 3 : connexion, exploration SQL, chargement et découpage
  02_modelisation.ipynb   # tâches 4 à 8 : fuite de données, v1 rejouée, pipeline v2, réglage, verdict
  essais_hyperparametres.csv  # trace des essais de réglage (tâche 7)
model/
  train_baseline.py       # recette de la v1, réutilisée pour la rejouer (tâche 5)
  cancellation_model_v2.joblib  # modèle v2, produit par train.py
  model_info_v2.json      # fiche du modèle v2, produite par train.py
features.py               # colonnes écartées, colonnes d'entrée, variables nouvelles (add_features)
train.py                  # script d'entraînement final (tâche 9)
RESULTATS.md              # démarche, fuite de données, tableau v1 / v2, recommandation
```

## Démarrage

```bash
cp .env.example .env
docker compose up -d
```

Au premier démarrage, PostgreSQL crée les tables et charge l'historique (environ une minute). Vérifie que la base est prête :

```bash
docker compose ps
```

Le service `db` doit être *healthy*. Ensuite, dans ton environnement virtuel :

```bash
pip install -r requirements.txt
```

Puis ouvre `notebooks/01_exploration.ipynb` (dans VS Code ou Jupyter) et exécute les premières cellules.

Pour arrêter la base : `docker compose down`. Les données sont conservées. Pour **repartir d'une base neuve** (le chargement se refait au démarrage suivant) : `docker compose down -v`.

## Ré-entraîner le modèle

La base doit être démarrée et *healthy*.

```bash
python train.py
```

Le script charge les réservations par une requête SQL, met de côté les 4 derniers mois d'arrivée, valide le pipeline par validation croisée temporelle, l'entraîne et l'évalue sur le jeu de test. Il écrit `model/cancellation_model_v2.joblib` et `model/model_info_v2.json`. Les résultats sont dans [RESULTATS.md](RESULTATS.md).

Pour charger le modèle ailleurs (API), `features.py` doit être importable : le modèle enregistré référence la fonction `add_features`.

## La base

**`hotels`** — les deux établissements.

| Colonne | Contenu |
|---|---|
| `hotel_id` | identifiant |
| `code` | `City Hotel` ou `Resort Hotel` |
| `nom`, `type`, `ville` | description de l'établissement |

**`reservations`** — une ligne par réservation, `hotel_id` renvoie vers `hotels`. Les principales colonnes :

| Colonne | Contenu |
|---|---|
| `is_canceled` | 1 si la réservation a été annulée (**la cible**) |
| `lead_time` | nombre de jours entre la réservation et la date d'arrivée prévue |
| `arrival_date` | date d'arrivée prévue (aussi décomposée en `arrival_date_year`, `_month`, `_week_number`, `_day_of_month`) |
| `stays_in_weekend_nights`, `stays_in_week_nights` | nuits réservées en week-end / en semaine |
| `adults`, `children`, `babies` | composition du groupe |
| `meal` | formule repas (`BB` petit-déjeuner, `HB` demi-pension, `FB` pension complète, `SC`/`Undefined` sans repas) |
| `country` | pays d'origine du client (code ISO) |
| `market_segment`, `distribution_channel` | segment de clientèle, canal de réservation (`TA` agence de voyage, `TO` voyagiste) |
| `is_repeated_guest`, `previous_cancellations`, `previous_bookings_not_canceled` | historique du client |
| `reserved_room_type`, `assigned_room_type` | type de chambre réservé / finalement attribué |
| `booking_changes` | nombre de modifications de la réservation |
| `deposit_type` | type d'acompte (`No Deposit`, `Refundable`, `Non Refund`) |
| `agent`, `company` | identifiant de l'agence ou de l'entreprise ayant réservé |
| `days_in_waiting_list` | jours passés en liste d'attente avant confirmation |
| `customer_type` | type de client (individuel, groupe, contrat…) |
| `adr` | prix moyen par nuit (*Average Daily Rate*), en € |
| `required_car_parking_spaces`, `total_of_special_requests` | places de parking, demandes particulières |
| `reservation_status`, `reservation_status_date` | dernier statut de la réservation et date de ce statut |

## Données

Extrait de *Hotel Booking Demand* — N. Antonio, A. de Almeida, L. Nunes (2019), *Data in Brief* 22, 41-49. Licence CC BY 4.0. Les noms des établissements sont fictifs.
