"""
Etapa 5 (v2): Inferencia en tiempo real, con landmarks normalizados.
Carga el modelo entrenado y predice la letra en vivo, usando la
misma normalizacion que se uso al recolectar el dataset.
"""

import pickle

import cv2
import mediapipe as mp

from utils import normalize_landmarks

MODEL_PATH = "models/sign_classifier.pkl"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def main():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo acceder a la webcam.")
        return

    print("Camara abierta. Presiona 'q' para salir.")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            prediction_text = "Sin mano detectada"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                features = normalize_landmarks(hand_landmarks)
                prediction = model.predict([features])[0]

                probabilities = model.predict_proba([features])[0]
                confidence = max(probabilities)

                prediction_text = f"Letra: {prediction} ({confidence:.0%})"

            cv2.putText(
                frame, prediction_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )
            cv2.imshow("Etapa 5 - Inferencia en tiempo real", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()