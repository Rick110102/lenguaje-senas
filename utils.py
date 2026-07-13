"""
Utilidades compartidas: normalizacion de landmarks de la mano.

Convierte las coordenadas absolutas que da MediaPipe en coordenadas
relativas a la muneca y escaladas por el tamano de la mano, para que
el modelo aprenda la FORMA de la seña y no la posicion en el frame.
"""

import math


def normalize_landmarks(hand_landmarks):
    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Trasladar: la muneca (landmark 0) pasa a ser el origen (0, 0, 0)
    wx, wy, wz = coords[0]
    translated = [(x - wx, y - wy, z - wz) for x, y, z in coords]

    # Escalar: usamos la distancia muneca -> base del dedo medio (landmark 9)
    # como referencia del "tamano" de la mano en este frame
    rx, ry, rz = translated[9]
    scale = math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    if scale == 0:
        scale = 1e-6

    normalized = []
    for x, y, z in translated:
        normalized.extend([x / scale, y / scale, z / scale])

    return normalized