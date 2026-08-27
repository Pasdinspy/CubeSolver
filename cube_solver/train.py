"""
Entraine un modele YOLO (Ultralytics) a detecter les 6 couleurs de stickers
du cube a partir du dataset annote dans data/.

Pre-requis avant de lancer ce script:
  1. Avoir capture des images (cube_solver/capture_data.py).
  2. Avoir annote chaque image (bounding box par sticker + classe couleur)
     avec un outil type LabelImg / CVAT / Roboflow, au format YOLO.
  3. Avoir range les images/labels dans:
       data/images/train, data/labels/train
       data/images/val,   data/labels/val
  4. Verifier que data/dataset.yaml pointe vers ces dossiers et que les
     "names" correspondent a cube_solver/config.py::CLASS_NAMES.

Usage:
    python -m cube_solver.train --data data/dataset.yaml --epochs 100
"""

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement YOLO pour la detection des stickers du cube.")
    parser.add_argument("--data", type=str, default="data/dataset.yaml", help="Chemin du fichier dataset YOLO.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Poids de depart (backbone pre-entraine).")
    parser.add_argument("--epochs", type=int, default=100, help="Nombre d'epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Taille des images d'entrainement.")
    parser.add_argument("--batch", type=int, default=16, help="Taille de batch.")
    parser.add_argument("--project", type=str, default="models", help="Dossier de sortie des runs d'entrainement.")
    parser.add_argument("--name", type=str, default="cube_faces", help="Nom du run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    print(
        f"\nEntrainement termine. Le meilleur modele se trouve dans "
        f"{args.project}/{args.name}/weights/best.pt"
    )


if __name__ == "__main__":
    main()
