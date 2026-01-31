---
icon: lucide/fan
---

# Usage

Une segmentation avec le modèle SAM de l'urne s'effectue avec ces étapes

1. Lancer le programme `src/utils/redim_urn.py` pour créer une image rognée et
   redimensionnée de l'urne. Cette image sera cubique pour avoir des slices
   carrées pour le modèle SAM de segmentation.

    ```sh
    # --z, --y et --x correspondent aux segments rognés sur chaque axe
    # --out_size correspond à la taille de côté finale du cube redimensionné
    python src/utils/redim_urn.py --input data/<raw_tiff>.tiff \
    --output data/<resized_tiff>.tiff \
    --z 0 255 --y 0 255 --x 0 255 --out_size 128
    ```

2. Des analyses préliminaires des propriétés d'imagerie de l'urne peuvent être
   menées avec le programme `src/analyze_dataset_metrics.py` :

    ```sh
    python src/analyse_dataset_metrics.py --tiff data/<your_tiff_file>.tiff
    ```

    Les résultats sont disponibles dans `./ssim_results`

3. Configurer la méthode de chargement en mémoire de l'urne et les
   hyperparamètres de segmentation 3D en éditant ces fichiers `.yaml` :

    - `configs/data/urn.yaml`
    - `configs/model/sam.yaml`

    D'autres configurations peuvent être établies avec hydra. Une description
    introductive des hyperparamètres est consultable [ici](./config.md).

4. Exécuter l'évaluation avec le programme `src/eval.py`.

    ```sh
    # ajouter trainer=gpu pour avoir l'usage du gpu
    python src/eval.py trainer=gpu
    ```

5. Visualiser les résultats :

    1. La segmentation est dans un fichier tiff dans
       `logs/eval/runs/<timestamp_de_lexperience>/sam_urn_segmentation.tiff`
    2. Les métriques d'évaluation sont consultables dans l'expérience associées
       depuis l'interface de MLFlow. Pour ouvrir MLFlow

        ```sh
         cd logs/mlflow
         mlflow ui
        ```
