"""
Detector de velocidad vehicular en tiempo real usando OpenCV.

Uso:
    python detector_velocidad.py --video ruta/al/video.mp4
    python detector_velocidad.py --video ruta/al/video.mp4 --metros_por_pixel 0.05
    python detector_velocidad.py --video ruta/al/video.mp4 --mostrar_mascara

Controles:
    q / ESC  - Salir
    p        - Pausar / reanudar
    r        - Reiniciar video
    +/-      - Ajustar metros_por_pixel en tiempo real
"""

import argparse
import sys
import time

import cv2
import numpy as np


# ============================================================
# MÓDULO 1: CONFIGURACIÓN
# ============================================================

CONFIG = {
    "metros_por_pixel": 0.05,
    "min_area_contorno": 800,
    "max_area_contorno": 50000,
    "distancia_max_asociacion": 80,
    "frames_para_eliminar": 15,
    "historia_sustractor": 500,
    "umbral_sustractor": 50,
    "detectar_sombras": False,
    "kernel_morfologico": (5, 5),
    "grosor_bbox": 2,
    "color_bbox": (0, 255, 0),
    "color_velocidad": (0, 255, 255),
    "color_id": (255, 180, 0),
    "color_rastro": (0, 200, 255),
    "fuente": cv2.FONT_HERSHEY_SIMPLEX,
    "escala_fuente": 0.6,
    "grosor_texto": 2,
}


# ============================================================
# MÓDULO 2: SUSTRACCIÓN DE FONDO Y DETECCIÓN
# ============================================================

class DetectorVehiculos:
    def __init__(self, config):
        self.sustractor = cv2.createBackgroundSubtractorMOG2(
            history=config["historia_sustractor"],
            varThreshold=config["umbral_sustractor"],
            detectShadows=config["detectar_sombras"],
        )
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config["kernel_morfologico"]
        )
        self.min_area = config["min_area_contorno"]
        self.max_area = config["max_area_contorno"]

    def detectar(self, frame):
        mascara = self.sustractor.apply(frame)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self.kernel, iterations=2)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, self.kernel, iterations=3)
        mascara = cv2.dilate(mascara, self.kernel, iterations=1)

        contornos, _ = cv2.findContours(
            mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detecciones = []
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contorno)
                cx, cy = x + w // 2, y + h // 2
                detecciones.append((x, y, w, h, cx, cy))

        return detecciones, mascara


# ============================================================
# MÓDULO 3: MOTOR MATEMÁTICO (CINEMÁTICA)
# ============================================================

def calcular_velocidad_instantanea(track, dt, metros_por_pixel):
    if len(track) < 5:
        return 0
    p1, p2 = track[-5], track[-1]
    dx = (p2[0] - p1[0]) * metros_por_pixel
    dy = (p2[1] - p1[1]) * metros_por_pixel
    vx, vy = dx / (5 * dt), dy / (5 * dt)
    return np.sqrt(vx**2 + vy**2) * 3.6


# ============================================================
# MÓDULO 4: SEGUIMIENTO (TRACKER)
# ============================================================

class Vehiculo:
    _siguiente_id = 1

    def __init__(self, cx, cy, bbox):
        self.id = Vehiculo._siguiente_id
        Vehiculo._siguiente_id += 1
        self.posiciones = [(cx, cy)]
        self.bbox = bbox
        self.frames_sin_ver = 0
        self.velocidad_kmh = 0.0

    def actualizar(self, cx, cy, bbox, dt, metros_por_pixel):
        self.posiciones.append((cx, cy))
        if len(self.posiciones) > 30:
            self.posiciones.pop(0)
        self.bbox = bbox
        self.frames_sin_ver = 0
        self.velocidad_kmh = calcular_velocidad_instantanea(
            self.posiciones, dt, metros_por_pixel
        )

    @property
    def centro(self):
        return self.posiciones[-1]


class TrackerVehiculos:
    def __init__(self, config):
        self.vehiculos = []
        self.dist_max = config["distancia_max_asociacion"]
        self.frames_eliminar = config["frames_para_eliminar"]

    def actualizar(self, detecciones, dt, metros_por_pixel):
        if not self.vehiculos:
            for (x, y, w, h, cx, cy) in detecciones:
                self.vehiculos.append(Vehiculo(cx, cy, (x, y, w, h)))
            return self.vehiculos

        usados_vehiculos = set()
        usados_detecciones = set()

        costos = []
        for i, vehiculo in enumerate(self.vehiculos):
            for j, (x, y, w, h, cx, cy) in enumerate(detecciones):
                dist = np.sqrt(
                    (vehiculo.centro[0] - cx) ** 2
                    + (vehiculo.centro[1] - cy) ** 2
                )
                if dist < self.dist_max:
                    costos.append((dist, i, j))

        costos.sort()
        for dist, i, j in costos:
            if i in usados_vehiculos or j in usados_detecciones:
                continue
            x, y, w, h, cx, cy = detecciones[j]
            self.vehiculos[i].actualizar(cx, cy, (x, y, w, h), dt, metros_por_pixel)
            usados_vehiculos.add(i)
            usados_detecciones.add(j)

        for j, (x, y, w, h, cx, cy) in enumerate(detecciones):
            if j not in usados_detecciones:
                self.vehiculos.append(Vehiculo(cx, cy, (x, y, w, h)))

        for i, vehiculo in enumerate(self.vehiculos):
            if i not in usados_vehiculos:
                vehiculo.frames_sin_ver += 1

        self.vehiculos = [
            v for v in self.vehiculos if v.frames_sin_ver < self.frames_eliminar
        ]

        return self.vehiculos


# ============================================================
# MÓDULO 5: VISUALIZACIÓN
# ============================================================

class Visualizador:
    def __init__(self, config):
        self.cfg = config

    def dibujar(self, frame, vehiculos):
        for v in vehiculos:
            if v.frames_sin_ver > 0:
                continue

            x, y, w, h = v.bbox

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                self.cfg["color_bbox"],
                self.cfg["grosor_bbox"],
            )

            if len(v.posiciones) > 1:
                pts = np.array(v.posiciones, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, self.cfg["color_rastro"], 2)

            texto_id = f"ID:{v.id}"
            cv2.putText(
                frame,
                texto_id,
                (x, y - 30),
                self.cfg["fuente"],
                self.cfg["escala_fuente"] * 0.8,
                self.cfg["color_id"],
                1,
            )

            if v.velocidad_kmh > 2:
                texto_vel = f"{v.velocidad_kmh:.1f} km/h"
                (tw, th), _ = cv2.getTextSize(
                    texto_vel,
                    self.cfg["fuente"],
                    self.cfg["escala_fuente"],
                    self.cfg["grosor_texto"],
                )
                cv2.rectangle(
                    frame,
                    (x, y - 25),
                    (x + tw + 8, y - 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    texto_vel,
                    (x + 4, y - 10),
                    self.cfg["fuente"],
                    self.cfg["escala_fuente"],
                    self.cfg["color_velocidad"],
                    self.cfg["grosor_texto"],
                )

        return frame

    def dibujar_info(self, frame, fps, num_vehiculos, metros_por_pixel):
        h, w = frame.shape[:2]
        info = [
            f"FPS: {fps:.0f}",
            f"Vehiculos: {num_vehiculos}",
            f"Escala: {metros_por_pixel:.4f} m/px",
            "q:Salir  p:Pausa  r:Reiniciar  +/-:Escala",
        ]
        y_offset = 25
        for linea in info:
            cv2.putText(
                frame, linea, (10, y_offset),
                self.cfg["fuente"], 0.5, (255, 255, 255), 1,
            )
            y_offset += 22
        return frame


# ============================================================
# MÓDULO 6: APLICACIÓN PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Detector de velocidad vehicular en tiempo real"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Ruta al archivo de video",
    )
    parser.add_argument(
        "--metros_por_pixel", type=float, default=CONFIG["metros_por_pixel"],
        help=f"Factor de conversión metros/pixel (default: {CONFIG['metros_por_pixel']})",
    )
    parser.add_argument(
        "--mostrar_mascara", action="store_true",
        help="Mostrar ventana con la máscara de sustracción de fondo",
    )
    parser.add_argument(
        "--guardar", type=str, default=None,
        help="Ruta para guardar el video procesado (ej: salida.avi)",
    )
    args = parser.parse_args()

    CONFIG["metros_por_pixel"] = args.metros_por_pixel

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video '{args.video}'")
        sys.exit(1)

    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps_video
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {args.video}")
    print(f"Resolución: {ancho}x{alto} | FPS: {fps_video:.1f} | Frames: {total_frames}")
    print(f"Escala: {CONFIG['metros_por_pixel']} metros/pixel")
    print("Controles: q=Salir, p=Pausa, r=Reiniciar, +/-=Ajustar escala")

    detector = DetectorVehiculos(CONFIG)
    tracker = TrackerVehiculos(CONFIG)
    visualizador = Visualizador(CONFIG)

    escritor = None
    if args.guardar:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        escritor = cv2.VideoWriter(args.guardar, fourcc, fps_video, (ancho, alto))
        print(f"Guardando video procesado en: {args.guardar}")

    pausado = False
    frame_num = 0
    t_anterior = time.time()

    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            frame_num += 1

            t_actual = time.time()
            fps_real = 1.0 / max(t_actual - t_anterior, 1e-6)
            t_anterior = t_actual

            detecciones, mascara = detector.detectar(frame)

            vehiculos = tracker.actualizar(
                detecciones, dt, CONFIG["metros_por_pixel"]
            )

            frame = visualizador.dibujar(frame, vehiculos)
            num_activos = sum(1 for v in vehiculos if v.frames_sin_ver == 0)
            frame = visualizador.dibujar_info(
                frame, fps_real, num_activos, CONFIG["metros_por_pixel"]
            )

            cv2.imshow("Detector de Velocidad Vehicular", frame)

            if args.mostrar_mascara:
                cv2.imshow("Mascara", mascara)

            if escritor:
                escritor.write(frame)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (ord("q"), 27):
            break
        elif tecla == ord("p"):
            pausado = not pausado
            print("Pausado" if pausado else "Reanudado")
        elif tecla == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = TrackerVehiculos(CONFIG)
            Vehiculo._siguiente_id = 1
            frame_num = 0
            print("Video reiniciado")
        elif tecla == ord("+") or tecla == ord("="):
            CONFIG["metros_por_pixel"] *= 1.1
            print(f"Escala: {CONFIG['metros_por_pixel']:.4f} m/px")
        elif tecla == ord("-"):
            CONFIG["metros_por_pixel"] /= 1.1
            print(f"Escala: {CONFIG['metros_por_pixel']:.4f} m/px")

    cap.release()
    if escritor:
        escritor.release()
    cv2.destroyAllWindows()
    print(f"Procesados {frame_num} frames.")


if __name__ == "__main__":
    main()
