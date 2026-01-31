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
    # in the configs/experiment/example.yaml
    # use 25D encoding with a jump of k slices, with k sweeped
    data:
      slice_batch_size: 20
      use_25d_image: True

    # here, data.slice_jump undergoes a value sweeping
    sweeper:
      params:
        data:
          slice_jump: 3,5,15
    ```

    ```sh
    python src/eval.py +experiment=example
    ```

## Configuration de l'encodage de l'image pour SAM

Elle est définie dans 
