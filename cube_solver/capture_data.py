"""
Outil de collecte d'images pour constituer le dataset d'entrainement YOLO.

Ouvre la webcam et permet de sauvegarder des photos d'une face du cube.
Ces images devront ensuite etre annotees (bounding box + couleur) avec un
outil externe (LabelImg, Roboflow, CVAT, ...) au format YOLO, puis placees
dans data/images/train (ou val) avec leur fichier .txt correspondant dans
data/labels/train (ou val).

Usage:
    python -m cube_solver.capture_data --camera 0 --out data/raw

Controles:
    [espace] ou [c] : capturer une image
    [q] ou [Echap]  : quitter
"""

import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collecte d'images du cube pour l'entrainement YOLO.")
    parser.add_argument("--camera", type=int, default=0, help="Index de la webcam (defaut: 0).")
    parser.add_argument("--out", type=str, default="data/raw", help="Dossier de sortie des images capturees.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la camera d'index {args.camera}.")

    print("Camera ouverte. [espace/c] pour capturer, [q/Echap] pour quitter.")
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Erreur de lecture de la webcam.")
                break

            preview = frame.copy()
            cv2.putText(
                preview,
                f"Images capturees: {count}  |  [espace] capturer  [q] quitter",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Capture cube - face", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q ou Echap
                break
            if key in (ord("c"), 32):  # c ou espace
                filename = out_dir / f"cube_{int(time.time() * 1000)}.jpg"
                cv2.imwrite(str(filename), frame)
                count += 1
                print(f"Image sauvegardee: {filename}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Termine. {count} image(s) sauvegardee(s) dans {out_dir}.")


if __name__ == "__main__":
    main()
