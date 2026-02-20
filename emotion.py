import cv2
from deepface import DeepFace

def detect_emotion():
    cap = cv2.VideoCapture(0)
    print("Press Q to capture emotion")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Interview Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    result = DeepFace.analyze(frame, actions=['emotion'])
    return result[0]['dominant_emotion']