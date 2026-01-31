# Hyperparamétrage de l'expérience

Dans cette page, une introduction à l'usage d'Hydra va être établi pour décrire
comment hyperparamétrer chaque exécution d'une segmentation zero-shot.

## Comment hyperparamétrer ?

Hydra propose trois méthodes qui peuvent être combinées :

1. Éditer directement les fichiers de configuration par défaut utilisé par
   `src/eval.py`. Ces fichiers sont des `.yaml` situés dans `configs/*/`

2. Surcharger les configuration par défaut au moment de l'exécution de la
   commmande

    ```sh
    python src/eval.py data.filename=./data/my_tiff_file.tiff
    ```

3. Ajouter ces surcharges à un fichier d'expérience. Cette pratique est
   recommandé pour reproduire une surcharge d'hyperparamètre menant à une
segmentation intéressante. Il faut écrire ce fichier `.yaml` dans le dossier
`configs/experiment`. Voir le fichier `configs/experiment/example.yaml` qui
exécute une segmentation de l'image avec un encodage 2.5D et un slice jump de
3.

    ```sh
    # si le fichier s'appelle example.yaml
    python src/eval.py +experiment=example
    ```

## Exécuter plusieurs exécutions

Il est possible de plannifier plusieurs exécutions complètes de la segmentation
en faisant varier des hyperparamètres données. Deux méthodes possibles:

1. Depuis la ligne de commande :

    ```sh
    python src/eval.py -m data.slice_jump=3,5,15
    ```

2. Depuis un fichier de configuration (e.g. le fichier d'expérience), avec le
   paramètre `sweeper.params`

    ```yaml
    # dans le fichier configs/experiment/example.yaml
    # encode en 2.5D avec un saut de slices variant à chaque exécution
    data:
      slice_batch_size: 20
      use_25d_image: True

    # ici, le balayage de valeurs est configuré pour data.slice_jump
    sweeper:
      params:
        data:
          slice_jump: 3,5,15
    ```

    ```sh
    python src/eval.py +experiment=example
    ```

## Description d'hyperparamètres notoires

Les hyperparamètres importants sont des arguments de construction de classes
python instanciées à partir du bon fichier `.yaml` Le docstring du
constructeur de la classe dans le bon fichier python situé sans `src/`
constitue donc la meilleure référence.

### Configuration de l'encodage de l'image pour SAM

- Fichier de configuration : `configs/data/urn.yaml`
- Classe instanciée : `OneUrnDataModule`
- Module associé : `src/data/one_urn_datamodule.py`.

Les hyperparamètres suivants pourraient être intéressants à éditer :

- `filename` + `ground_truth_filename`: les chemins des fichiers `.tiff` de la
tomographie de l'urne et d'une vérité de terrain segmentée
- `slicing_axis`: l'axe normal aux slices à segmenter (`z`, `y` ou `x`)
- `use_25d_image` + `slice_jump`: le mode d'encodage, valant `False` s'il est
standard, `True` s'il est en 2.5D et `clahe` s'il est en 2.5D avec une
application du filtre CLAHE
- `slice_batch_size`: le nombre de slices à faire segmenter avant de mettre à
jour le fichier `.tiff` de sortie contenant la segmentation
- `slice_image_size`: un redimenssionnement supplémentaire des slices avant de
  les passer dans SAM

### Configuration zero-shot de SAM

Le modèle SAM implémenté par Ultralytics peut-être configuré depuis hydra.

- Fichier de configuration : `configs/model/sam.yaml`
- Classe instanciée : `Sam3DModuleLinear`
- Module associé : `src/model/sam_module.py`

Les hyperparamètres suivants pourraient être intéressants à éditer :

- `sam_overrides`: tous les hyperparamètres d'inférence du modèle implémenté
  par Ultralytics
- `points_batch_size`: le nombre de points de prompt traité par 1 inférence de
SAM. Il est possible que le nombre total de points de prompt soit trop
important et fasse saturer la mémoire utilisé par Ultralytics. Pour éviter
cela, plusieurs segmentations avec des sous-paquets de points de prompt sont
séquentiellement effectuées. Cet hyperparamère fixe cette taille de
sous-paquet. À ajuster en fonction du matériel.
- `infer`: la stratégie de prompt engineering appliquée à chaque slice pour
effectuer la segmentation. L'option `grid_stride` fixera par exemple la
résolution de la grille de points de prompt dans une stratégie de segmentation
non-guidée (`mode: grid`) et simplement définie depuis un maillage de l'image.

### Écriture de la segmentation sur disque

- Fichier de configuration : `configs/callbacks/tiff_writing.yaml`
- Classe instanciée : `SaveSegmentOnTheFly`
- Module associé : `src/utils/imageproc/save_on_disk.py`

Pour personnaliser le chemin du fichier de segmentation :

- `output_file`

### CodeCarbon

- Fichier de configuration : `configs/extras/default.yaml`
- Classe instanciée : `codecarbon.EmissionsTracker`
- Librairie associée : [`codecarbon`](https://mlco2.github.io/codecarbon/parameters)

### MLFLow

- Fichier de configuration : `configs/logger/mlflow.yaml`
- Classe instanciée : `MLFLowLogger`
- Librairie associée : [`Intégration de PyTorch Lightning`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.MLFlowLogger.html)

Vous souhaiteirez éditer ces hyperparamètres :

- `experiment_name` : nom de l'expérience dans l'interface de MLFlow
- `run_name`: nom de l'exécution au sein de l'expérience dans l'interface de
MLFlow
- `tags`
