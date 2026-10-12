"""
Herramienta de calibración para calcular metros_por_pixel.

Uso:
    python calibrar.py --video ruta/al/video.mp4

Instrucciones:
    1. Se mostrará el primer frame del video.
    2. Haz clic en dos puntos cuya distancia real (en metros) conozcas
       (por ejemplo, el ancho de un carril: ~3.5m).
    3. Ingresa la distancia real en metros cuando se te pida.
    4. El programa calculará metros_por_pixel.
"""

import argparse
import sys

import cv2
import numpy as np

puntos = []


def click_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 2:
        puntos.append((x, y))
        frame = param["frame"]
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(puntos) == 2:
            cv2.line(frame, puntos[0], puntos[1], (0, 255, 0), 2)
            dist_px = np.sqrt(
                (puntos[1][0] - puntos[0][0]) ** 2
                + (puntos[1][1] - puntos[0][1]) ** 2
            )
            cv2.putText(
                frame, f"{dist_px:.1f} px",
                ((puntos[0][0] + puntos[1][0]) // 2, (puntos[0][1] + puntos[1][1]) // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
            )
        cv2.imshow("Calibracion", frame)


def main():
    parser = argparse.ArgumentParser(description="Calibrar metros_por_pixel")
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir '{args.video}'")
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error al leer el primer frame.")
        sys.exit(1)

    datos = {"frame": frame.copy()}
    cv2.imshow("Calibracion", frame)
    cv2.setMouseCallback("Calibracion", click_handler, datos)

    print("Haz clic en DOS puntos cuya distancia real conozcas.")
    print("Presiona cualquier tecla cuando hayas marcado ambos puntos.")

    while True:
        key = cv2.waitKey(100) & 0xFF
        if len(puntos) == 2:
            break
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    dist_px = np.sqrt(
        (puntos[1][0] - puntos[0][0]) ** 2
        + (puntos[1][1] - puntos[0][1]) ** 2
    )

    print(f"\nDistancia en pixeles: {dist_px:.1f} px")
    try:
        dist_real = float(input("Ingresa la distancia real en metros: "))
    except ValueError:
        print("Valor inválido.")
        sys.exit(1)

    mpp = dist_real / dist_px
    print(f"\nmetros_por_pixel = {mpp:.6f}")
    print(f"\nUsa este valor así:")
    print(f"  python detector_velocidad.py --video {args.video} --metros_por_pixel {mpp:.6f}")


if __name__ == "__main__":
    main()
