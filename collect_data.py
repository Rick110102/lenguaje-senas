"""
Etapa 3 (v2): Construccion del dataset, con landmarks normalizados.
Recolecta landmarks de la mano para cada letra y los guarda
etiquetados en un archivo CSV, ya normalizados (posicion relativa
a la muneca, escalados por tamano de mano).
"""

import csv

import cv2
import mediapipe as mp

from utils import normalize_landmarks

LETTERS = ["A", "B", "C", "F", "I", "L", "O", "Y"]
SAMPLES_PER_LETTER = 60
OUTPUT_CSV = "dataset.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la webcam.")
        return

    # Empezamos el CSV desde cero para no mezclar datos viejos (sin
    # normalizar) con los nuevos (normalizados)
    csv_file = open(OUTPUT_CSV, mode="w", newline="")
    writer = csv.writer(csv_file)

    header = ["label"]
    for i in range(21):
        header.extend([f"x{i}", f"y{i}", f"z{i}"])
    writer.writerow(header)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        for letter in LETTERS:
            print(f"\nSiguiente letra: {letter}")
            print("Posiciona tu mano y presiona 's' para empezar a grabar.")

            waiting = True
            while waiting:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                cv2.putText(
                    frame, f"Letra: {letter} - presiona 's' para grabar",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                )
                cv2.imshow("Etapa 3 - Recoleccion de dataset", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    waiting = False
                elif key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    csv_file.close()
                    return

            collected = 0
            while collected < SAMPLES_PER_LETTER:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    features = normalize_landmarks(hand_landmarks)
                    writer.writerow([letter] + features)
                    collected += 1

                cv2.putText(
                    frame, f"Grabando '{letter}': {collected}/{SAMPLES_PER_LETTER}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, "Mueve un poco la mano: angulo, distancia, posicion",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
                )
                cv2.imshow("Etapa 3 - Recoleccion de dataset", frame)
                cv2.waitKey(1)

            print(f"Letra '{letter}' completa: {collected} muestras guardadas.")

    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset guardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()