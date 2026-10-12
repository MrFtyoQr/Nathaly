# Detector de Velocidad Vehicular

Sistema de detección y medición de velocidad de vehículos en tiempo real usando OpenCV.

## Requisitos

```bash
pip install -r requirements.txt
```

- Python 3.8+
- OpenCV 4.8+
- NumPy

## Uso

### Detectar velocidad en un video

```bash
python detector_velocidad.py --video mi_video.mp4
```

### Con escala personalizada

```bash
python detector_velocidad.py --video mi_video.mp4 --metros_por_pixel 0.03
```

### Guardar video procesado

```bash
python detector_velocidad.py --video mi_video.mp4 --guardar salida.avi
```

### Ver máscara de detección

```bash
python detector_velocidad.py --video mi_video.mp4 --mostrar_mascara
```

## Calibración

Para obtener mediciones precisas, calibra `metros_por_pixel` con tu video:

```bash
python calibrar.py --video mi_video.mp4
```

1. Marca dos puntos cuya distancia real conozcas (ancho de carril ~3.5m, largo de un carro ~4.5m).
2. Ingresa la distancia en metros.
3. Usa el valor calculado con `--metros_por_pixel`.

## Controles

| Tecla | Acción |
|-------|--------|
| `q` / `ESC` | Salir |
| `p` | Pausar / reanudar |
| `r` | Reiniciar video |
| `+` | Aumentar escala 10% |
| `-` | Reducir escala 10% |

## Cómo funciona

1. **Sustracción de fondo** (MOG2) para aislar objetos en movimiento.
2. **Operaciones morfológicas** para limpiar ruido.
3. **Detección de contornos** para identificar vehículos.
4. **Tracking por proximidad** para seguir cada vehículo entre frames.
5. **Cálculo cinemático** de velocidad instantánea (km/h) usando los últimos 5 puntos del track.
