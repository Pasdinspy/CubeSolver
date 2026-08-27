"""
Reconnaissance d'une face du cube: detecte les 9 stickers avec le modele
YOLO entraine (cube_solver/train.py) et affiche/renvoie la grille 3x3 de
couleurs correspondante.

Usage:
    # Webcam en continu, affiche la grille des qu'une face complete (9
    # stickers) est detectee.
    python -m cube_solver.detect_face --weights models/cube_faces/weights/best.pt --source 0

    # Une seule image.
    python -m cube_solver.detect_face --weights models/cube_faces/weights/best.pt --source chemin/vers/image.jpg
"""

import argparse

import cv2
from ultralytics import YOLO

from cube_solver.config import DRAW_COLORS
from cube_solver.grid_utils import Detection, detections_to_grid, grid_to_string


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection d'une face du cube et extraction de sa grille de couleurs.")
    parser.add_argument("--weights", type=str, required=True, help="Chemin du modele YOLO entraine (best.pt).")
    parser.add_argument("--source", type=str, default="0", help="Index webcam (ex: 0) ou chemin d'une image.")
    parser.add_argument("--conf", type=float, default=0.5, help="Seuil de confiance minimum.")
    return parser.parse_args()


def run_inference(model: YOLO, frame, conf: float):
    results = model.predict(frame, conf=conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        confidence = float(box.conf[0].item())
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        detections.append(Detection(x_center, y_center, label, confidence))

        color = DRAW_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            frame,
            f"{label} {confidence:.2f}",
            (int(x1), max(int(y1) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return detections, frame


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)

    # Source webcam si --source est un entier, sinon on traite comme une image.
    is_webcam = args.source.isdigit()

    if is_webcam:
        cap = cv2.VideoCapture(int(args.source))
        if not cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la camera d'index {args.source}.")

        print("Camera ouverte. [q/Echap] pour quitter.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erreur de lecture de la webcam.")
                    break

                detections, annotated = run_inference(model, frame, args.conf)
                grid = detections_to_grid(detections)

                status = f"Stickers detectes: {len(detections)}/9"
                cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Reconnaissance face du cube", annotated)

                if grid is not None:
                    print("\nFace detectee:")
                    print(grid_to_string(grid))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(f"Image introuvable: {args.source}")

        detections, annotated = run_inference(model, frame, args.conf)
        grid = detections_to_grid(detections)

        if grid is None:
            print(f"Attention: {len(detections)} sticker(s) detecte(s) au lieu de 9. Grille incomplete.")
        else:
            print("Face detectee:")
            print(grid_to_string(grid))

        cv2.imshow("Reconnaissance face du cube", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
