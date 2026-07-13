import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la webcam.")
        return

    print("Camara abierta. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Etapa 1 - Captura de video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()