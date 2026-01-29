# Installation avec Pixi

Pixi est l'outil recommandé. En effet,

- il reproduit un environnement complet (avec la bonne version de Python) et
  disponible sur tous les systèmes
- l'installation de PyTorch et Ultralytics s'adapte automatiquement à la
  disponibilité ou l'absence des drivers CUDA (pour exécuter les calculs sur GPU
  ou non)
- l'environnement est installé localement est très facilement retirable
- l'outil d'installation est plus rapide que dans toutes les autres méthodes

Pour l'installer, un script d'installation peu invasif peut être récupéré
depuis [cette page](https://pixi.prefix.dev/latest/installation/).

## Installer l'espace de travail

```sh
# à la racine du dépôt
pixi install
```

## Entrer dans l'environnement d'exécution

```sh
pixi shell
```

## Entrer dans un environnement de développement

Si du code doit être édité, un environnement de développement est proposé avec
des outils supplémentaires (formatter, pre-commits, builder de documentation)
qui seront installés.

```sh
pixi shell -e dev
```

## Entrer dans l'environnement pour napari

```sh
pixi shell -e napari
```
