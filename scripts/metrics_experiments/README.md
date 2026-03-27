# Expérimentations SAM pour Segmentation 3D

Ce dossier contient les expériences de segmentation utilisant SAM (Segment Anything Model) sur des images 3D.

## Structure du projet

```
.
├── config.py                      # Configuration centralisée
├── utils.py                       # Fonctions utilitaires
├── sam_inference.py               # Fonctions d'inférence SAM
├── exp1_grid_vs_boxes.py          # Expérience: Grid Points vs Bounding Boxes
├── exp2_yolo_detection.py         # Expérience: Détection YOLO
├── exp3_25d_segmentation.py       # Expérience: Segmentation 2.5D
├── exp4_iou_comparison.py         # Expérience: Comparaison globale
├── requirements.txt               # Dépendances Python
└── README.md                      # Ce fichier
```

## Installation

### Prérequis
- Python 3.8+
- CUDA 11.0+ (recommandé pour GPU)

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

Avant de lancer les expériences, modifiez le fichier `config.py` selon vos besoins :

```python
class Config:
    # Chemins
    TIF_PATH = '../tests/Romane_Martin_urne_sature_10-4.tif'  # Votre fichier .tif
    SAM_WEIGHTS = "sam_b.pt"                                   # Poids SAM
    YOLO_WEIGHTS = "yolo11n.pt"                                # Poids YOLO
    OUTPUT_DIR = "experiments_results"                          # Dossier de sortie
    
    # Paramètres SAM
    GRID_STRIDE = 64        # Densité des points de grille
    MIN_AREA = 300          # Aire minimale d'un masque (pixels)
    CONF_THR = 0.5          # Seuil de confiance SAM
    
    # Paramètres prétraitement
    URNA_THRESHOLD = 60     # Seuil de détection de l'urne
    USE_CLAHE = True        # Activer CLAHE pour améliorer contraste
```

## 🧪 Expériences

### Expérience 1 : Grid Points vs Bounding Boxes

Compare les performances de segmentation entre prompts de type "points en grille" et "bounding boxes".

**Lancement :**
```bash
python exp1_grid_vs_boxes.py
```

**Résultats :**
- Graphique d'évolution IoU entre slices consécutives
- Moyennes IoU pour chaque méthode
- Fichier `results.npz` avec données brutes

**Métrique :** IoU (Intersection over Union) entre masques de slices adjacentes

---

### Expérience 2 : Détection YOLO

Teste l'utilisation de YOLO pour détecter automatiquement la région d'intérêt (urne) avant segmentation SAM.

**Lancement :**
```bash
python exp2_yolo_detection.py
```

**Méthodes testées :**
1. YOLO seul
2. Seuillage seul
3. YOLO + Seuillage combinés

**Résultats :**
- Visualisation des 3 méthodes
- Analyse IoU avec YOLO + seuillage
- Nombre de masques détectés par méthode

---

### Expérience 3 : Segmentation 2.5D

Exploite l'information 3D en utilisant les slices adjacentes comme contexte dans les canaux RGB.

**Lancement :**
```bash
python exp3_25d_segmentation.py
```

**Tests effectués :**
1. Comparaison 2D classique vs 2.5D
2. Impact du paramètre `delta_z` (espacement entre slices)
3. Analyse IoU longitudinale (slice fixe vs slices suivantes)

**Résultats :**
- Graphiques 2D vs 2.5D
- Impact de delta_z sur nombre et aire des masques
- Évolution IoU

---

### Expérience 4 : Comparaison globale

Compare toutes les méthodes sur les mêmes données pour identifier la meilleure approche.

**Lancement :**
```bash
python exp4_iou_comparison.py
```

**Méthodes comparées :**
1. Grid Points
2. Bounding Boxes
3. YOLO + Threshold
4. 2.5D

**Résultats :**
- Graphique comparatif de toutes les méthodes
- Barres des moyennes IoU
- Tableau récapitulatif CSV
- Recommandation de la meilleure méthode

---

## Interprétation des résultats

### IoU (Intersection over Union)
- **IoU > 0.7** : Excellente stabilité entre slices
- **0.5 < IoU < 0.7** : Bonne stabilité
- **IoU < 0.5** : Segmentation instable

### Nombre de masques
- Idéalement : stable entre slices adjacentes
- Variation importante = segmentation peu robuste


## Analyse typique des résultats

D'après les expériences, on observe généralement :

1. **Grid Points** : Bon compromis performance/stabilité
2. **Bounding Boxes** : Performances similaires, parfois plus stable
3. **YOLO + Threshold** : Améliore la localisation mais peut être plus coûteux
4. **2.5D** : Meilleure utilisation du contexte 3D, recommandé si ressources disponibles

## Personnalisation

### Ajouter une nouvelle expérience

1. Créer un nouveau fichier `exp5_mon_experience.py`
2. Importer les modules nécessaires :
```python
from config import Config
from utils import *
from sam_inference import sam_inference
```
3. Implémenter votre logique
4. Sauvegarder les résultats dans `Config.OUTPUT_DIR`

### Modifier les paramètres SAM

Éditer `config.py` ou passer les paramètres directement :

```python
masks, colored, info = sam_inference(
    model,
    img_rgb=img,
    urna_mask=mask,
    mode="grid",
    grid_stride=32,      # Plus dense
    conf=0.7,            # Plus strict
    min_area=500         # Masques plus grands
)
```

