"""
Regroupe les detections YOLO (une par sticker) en une grille 3x3 ordonnee
(ligne par ligne, de haut en bas et de gauche a droite).
"""

from dataclasses import dataclass
from typing import List, Optional

from cube_solver.config import GRID_SIZE, STICKERS_PER_FACE


@dataclass
class Detection:
    x_center: float
    y_center: float
    label: str
    confidence: float


def detections_to_grid(detections: List[Detection]) -> Optional[List[List[str]]]:
    """
    Convertit une liste de detections (centre + label couleur) en grille 3x3.

    Retourne None si le nombre de detections n'est pas egal a 9 (une face
    complete), auquel cas l'appelant doit redemander une capture.
    """
    if len(detections) != STICKERS_PER_FACE:
        return None

    # Etape 1: regrouper les stickers par ligne en clusterisant sur y.
    sorted_by_y = sorted(detections, key=lambda d: d.y_center)
    rows = [sorted_by_y[i : i + GRID_SIZE] for i in range(0, STICKERS_PER_FACE, GRID_SIZE)]

    # Etape 2: dans chaque ligne, trier par x pour avoir gauche -> droite.
    grid: List[List[str]] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda d: d.x_center)
        grid.append([d.label for d in row_sorted])

    return grid


def grid_to_string(grid: List[List[str]]) -> str:
    return "\n".join(" | ".join(f"{cell:>7}" for cell in row) for row in grid)
