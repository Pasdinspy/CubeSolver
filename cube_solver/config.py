# Classes du modele YOLO : une classe par couleur de sticker.
# L'ordre ici doit correspondre EXACTEMENT a l'ordre utilise lors de l'annotation
# et dans data/dataset.yaml (champ "names").
CLASS_NAMES = ["white", "yellow", "red", "orange", "blue", "green"]

# Couleur BGR (OpenCV) utilisee pour dessiner chaque classe a l'ecran.
DRAW_COLORS = {
    "white": (255, 255, 255),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
}

# Nombre de stickers attendus sur une face du cube.
STICKERS_PER_FACE = 9
GRID_SIZE = 3
