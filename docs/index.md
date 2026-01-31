---
icon: lucide/house
---

# FUNERIA

FUNERIA est un projet expérimental de mise en application de technologies de
deep learning sur une tâche de segmentation 3D de vestiges d'urnes funéraires.

Une spécificité de ce projet aura été d'utiliser le modèle SAM de Meta avec un
jeu de données limité favorisant une première approche zero-shot suivie de
tentatives de finetuning.

L'approche zero-shot est intégrée dans un écosystème utilisant Hydra pour
configurer des expériences d'évaluation. D'autres contributions comme une
approche de finetuning ont été programmée dans un script extérieur par manque
de temps pour une intégration dans l'écosystème.

## Plan de la documentation

Cette documentation contient :

1. Les instructions pour installer l'environnement d'expérimentation, avec
   notamment Ultralytics et PyTorch

2. Les instructions pour effectuer les segmentations supportées par
   l'écosystème Hydra/Lightning.

3. Des détails sur les hyperparamètres métiers ajustables pour exécuter les
   segmentatins zero-shot.

4. Des détails sur les autres contributions accessibles depuis le dépôt mais
   non-intégrées dans l'écosystème.

Le plupart des pages sont en français, ce projet ayant été bâti dans le
contexte d'une collaboration entre les établissements bordelais ENSEIRB-MATMECA
et ICMCB.
