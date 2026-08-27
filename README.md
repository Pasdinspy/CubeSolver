# CubeSolver

Solver de Rubik's Cube. **Etape actuelle : reconnaissance des faces du cube via YOLO.**

## Idee generale

Un modele YOLO est entraine a detecter chaque sticker d'une face du cube et a
le classer par couleur (`white`, `yellow`, `red`, `orange`, `blue`, `green`).
A partir des 9 detections d'une face, on reconstruit la grille 3x3 de
couleurs. Ce sera la brique de base du solver complet (les 6 faces scannees
donneront l'etat du cube a resoudre).

## Installation

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Collecter des images

Prends des photos de faces de cube avec la webcam (varie l'eclairage, l'angle,
le cube utilise) :

```bash
python -m cube_solver.capture_data --camera 0 --out data/raw
```

`espace`/`c` pour capturer, `q` pour quitter. Vise au moins 100-200 images
pour un premier modele correct.

### 2. Annoter les images

Chaque image doit etre annotee au format YOLO : une bounding box par sticker
visible, avec sa classe de couleur. Utilise un outil externe, par exemple :

- [LabelImg](https://github.com/HumanSignal/labelImg) (local, simple)
- [CVAT](https://www.cvat.ai/) ou [Roboflow](https://roboflow.com/) (en ligne)

Les classes doivent etre saisies **dans cet ordre** (voir
`cube_solver/config.py::CLASS_NAMES`) :

```
0: white
1: yellow
2: red
3: orange
4: blue
5: green
```

Repartis ensuite images + labels dans :

```
data/images/train/  data/labels/train/
data/images/val/    data/labels/val/
```

(chaque image `xxx.jpg` doit avoir son fichier `xxx.txt` du meme nom dans le
dossier `labels` correspondant). Une repartition 80/20 train/val est un bon
depart.

### 3. Entrainer le modele

```bash
python -m cube_solver.train --data data/dataset.yaml --epochs 100
```

Le meilleur modele est sauvegarde dans
`models/cube_faces/weights/best.pt`.

### 4. Reconnaitre une face

Webcam en continu :

```bash
python -m cube_solver.detect_face --weights models/cube_faces/weights/best.pt --source 0
```

Des que 9 stickers sont detectes, la grille 3x3 de couleurs de la face
s'affiche dans la console.

Sur une image unique :

```bash
python -m cube_solver.detect_face --weights models/cube_faces/weights/best.pt --source chemin/vers/image.jpg
```

## Structure du projet

```
cube_solver/
  config.py         # classes de couleurs, constantes
  capture_data.py    # collecte d'images webcam pour le dataset
  train.py            # entrainement YOLO (ultralytics)
  detect_face.py       # inference + reconstruction de la grille 3x3
  grid_utils.py        # regroupement des detections en grille 3x3
data/
  raw/               # images brutes capturees, avant annotation/tri
  dataset.yaml         # config du dataset YOLO
  images/{train,val}/  # images annotees
  labels/{train,val}/  # annotations YOLO (.txt)
models/                # poids entraines (sortie de train.py)
```

## Prochaines etapes (hors scope actuel)

- Scanner les 6 faces et assembler l'etat complet du cube.
- Implementer l'algorithme de resolution (ex: Kociemba).
- Afficher la solution (sequence de mouvements).
