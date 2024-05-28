import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Desactivar mensajes de información y advertencia de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desactivar las optimizaciones de oneDNN

import cv2
from fer import FER

# Función para analizar el video y almacenar las emociones detectadas
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_with_emotions = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar emociones en el frame
        emotion_detector = FER()
        emotions = emotion_detector.detect_emotions(frame)
        
        # Almacenar el frame y las emociones detectadas
        frames_with_emotions.append((frame, emotions))
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Procesados {frame_count} frames...")
    
    cap.release()
    return frames_with_emotions

# Función para mostrar los frames que contienen la emoción solicitada
def show_frames_by_emotion(frames_with_emotions, target_emotion):
    for frame, emotions in frames_with_emotions:
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            emotion_confidences = emotion['emotions']
            max_emotion = max(emotion_confidences, key=emotion_confidences.get)
            max_confidence = emotion_confidences[max_emotion]
            
            if max_emotion == target_emotion and max_confidence >= 0.6:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                emotion_text = f"{max_emotion}: {max_confidence:.2f}"
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                # Mostrar el frame
                cv2.imshow('Filtered Emotion Detection', frame)
                
                # Esperar brevemente para que el video sea visible
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

# Ruta al archivo de video
video_path = 'C:/Users/andre/Desktop/UMG/INTELIGENCIA ARTIFICIAL/video.mp4'

# Analizar el video y almacenar las emociones detectadas
print("Analizando video...")
frames_with_emotions = analyze_video(video_path)
print("Análisis completo.")

# Solicitar al usuario que ingrese una emoción
target_emotion = input("Ingrese una emoción (feliz, triste, enojo, asombro): ").strip().lower()

# Diccionario para mapear emociones en español a los nombres en inglés utilizados por FER
emotion_map = {
    "feliz": "happy",
    "triste": "sad",
    "enojo": "angry",
    "asombro": "surprise"
}

# Obtener la emoción en inglés correspondiente a la entrada del usuario
target_emotion_english = emotion_map.get(target_emotion, None)

if target_emotion_english:
    print(f"Mostrando frames con la emoción: {target_emotion} ({target_emotion_english})")
    # Mostrar los frames que contienen la emoción solicitada
    show_frames_by_emotion(frames_with_emotions, target_emotion_english)
else:
    print("Emoción no reconocida. Por favor, ingrese una de las siguientes: feliz, triste, enojo, asombro.")

# Cerrar todas las ventanas
cv2.destroyAllWindows()
