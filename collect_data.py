"""
Etapa 3: Construccion del dataset.
Recolecta landmarks de la mano para cada letra del alfabeto que
queremos reconocer, y los guarda etiquetados en un archivo CSV.
"""

import csv
import os

import cv2
import mediapipe as mp

# Letras a recolectar. Cambia esta lista si quieres otro set.
LETTERS = ["A", "B", "C", "F", "I", "L", "O", "Y"]

SAMPLES_PER_LETTER = 30
OUTPUT_CSV = "dataset.csv"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def landmarks_to_row(hand_landmarks, label):
    row = [label]
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la webcam.")
        return

    # Si el archivo no existe, escribimos el encabezado primero
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, mode="a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
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

            # Esperar a que el usuario presione 's' para esta letra
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

            # Grabar SAMPLES_PER_LETTER muestras para esta letra
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

                    row = landmarks_to_row(hand_landmarks, letter)
                    writer.writerow(row)
                    collected += 1

                cv2.putText(
                    frame, f"Grabando '{letter}': {collected}/{SAMPLES_PER_LETTER}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
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