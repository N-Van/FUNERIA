# Installation avec Conda

> L'environnement conda/mamba n'a pas d'avantages à être utilisé par
> rapport à la venv Python. Il est recommandé d'utiliser Pixi, qui est une
> alternative plus rapide.

Une installation avec `conda` ou `mamba` n'est restreinte que sur un
environnement avec cuda 13 dessus. Si la version disponible est différente,
penser à changer l'index-url dans le fichier `environment.yaml`

## Installer l'environnement

```sh
mamba env create -n FUNERIA -f environment.yaml
```

## Entrer dans l'environnement

```sh
mamba activate FUNERIA
```

## Installer napari

```sh
mamba create -n napari -f napari-env.yml
```

Pour exécuter napari

```sh
mamba activate napari
napari ....
```
