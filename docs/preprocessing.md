# Stratégies de Prétraitement et d'Inférence

## Redimensionnement Isotrope

Le volume brut de tomographie aux rayons X présente des dimensions non cubiques. L'application d'une opération de redimensionnement standard, telle que celle implémentée par défaut dans SAM, déformerait la géométrie de l'urne et altérerait la représentation des objets de l'urne.

Afin de préserver l'intégrité géométrique des données, on procède par un redimensionnement isotrope:

**Étape 1 : Cropping** – Élimination l'espace vide autour de l'urne afin de réduire le volume de données non pertinentes (l'arrière plan).

**Étape 2 : Padding** – Remplissage uniforme pour transformer le volume recadré en un cube.

**Étape 3 : Redimensionnement isotrope** – Application d'un facteur de redimensionnement identique sur les trois axes (X, Y, Z) pour atteindre les dimensions cibles (640×640×640 pixels) tout en maintenant les proportions originales de l'urne.



## Stratégies d'Empilement de Coupes

Le modèle SAM a été initialement conçu pour traiter des images en couleur à trois canaux (RGB). Or, notre volume est en niveaux de gris monocanal. Deux approches ont été explorées pour adapter ces données au format d'entrée attendu par SAM:

**Approche A : Duplication en niveaux de gris (2D)** – La coupe monocanal est simplement dupliquée sur les trois canaux RGB. Cette méthode triviale permet une compatibilité immédiate avec SAM, mais ne fournit aucune information contextuelle spatiale supplémentaire au modèle.

**Approche B : Empilement de coupes adjacentes (2.5D)** – Pour fournir au modèle un contexte spatial local, trois coupes consécutives sont empilées dans les trois canaux. Plus précisément, pour segmenter la coupe i, nous utilisons les coupes (i-1), i et (i+1) respectivement dans les canaux R, G et B. Cette approche enrichit l'information disponible en intégrant la continuité volumique.

Des variantes plus complexes de la méthode 2.5D ont également été testées afin d'évaluer l'impact de différentes stratégies de contexte :

**Empilement espacé (i ± Δz)** – Au lieu d'utiliser les coupes immédiatement adjacentes (i±1), nous avons testé des espacements plus larges (i±Δz).

**Empilement espacé avec fusion** – Plutôt que d'utiliser des coupes individuelles, nous avons fusionné de petites piles de coupes avant de les assigner aux canaux RGB. Plusieurs opérateurs de fusion ont été testés (moyenne, maximum, médiane). Cependant, ces méthodes ont conduit à une sur-détection d'objets comparativement à l'empilement simple, probablement en raison d'un lissage excessif ou d'une perte de détails fins.

En conclusion, l'approche 2.5D avec empilement de coupes adjacentes (i-1, i, i+1) ou avec un léger espacement (Δz=3) offre le meilleur compromis entre enrichissement du contexte spatial et préservation des détails structurels.


## Génération de Prompts et optimisation de calcul

L'objectif de cette stratégie est de contraindre SAM à générer la grille de points uniquement sur les régions pertinentes du volume, c'est-à-dire l'intérieur de l'urne. Sans cette contrainte, le modèle génère des prompts de segmentation de manière indiscriminée sur l'ensemble de l'image, incluant l'arrière-plan et produisant des masques parasites qui augmentent inutilement le coût de calcul et la complexité de l'inférence.

**Approche A : YOLO** – Une boîte englobante rectangulaire est détectée automatiquement à l'aide d'un modèle YOLO. Bien que cette approche soit rapide, la boîte englobante représente une approximation qui inclut inévitablement des zones d'arrière-plan, générant ainsi des prompts dans des régions non pertinentes.

**Approche B : Masquage binaire** – Un masque binaire est créé en extrayant le contour exact du contenu de l'urne. Les prompts de segmentation sont ensuite générés exclusivement à l'intérieur de ce contour, éliminant complètement les zones d'arrière-plan.

Les résultats empiriques confirment que le masquage binaire précis (Approche B) est nettement supérieur. Non seulement il réduit le nombre de faux positifs, mais il concentre également les capacités du modèle sur les objets archéologiques d'intérêt, rendant la pipeline de segmentation plus robuste et plus efficace.


