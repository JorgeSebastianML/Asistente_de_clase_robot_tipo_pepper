# Importar librerias
import cv2
import time
import warnings
# Importar clases propias
from Yolo.yolo_detection import Yolo_Detection
from Open_pose.pose_detection import Pose_Detection
from Kalman_filter.sort import Sort

# Desactivar los mensajes de warnings
warnings.filterwarnings("ignore")

# Funcion Main
def main(Camera = False):
    # Instanciar objeto Yolo, para realizar el reconocimiento de objetos y de poses
    Yolo = Yolo_Detection(use_gpu=True, confidence=0.45, threshold=0.3, size=608)
    # Instanciar objeto OpenPose para la recontrucion de poses
    OpenPose = Pose_Detection()
    # Instanciar objeto sort para el uso del filtro de kalman
    sort = Sort()
    # Instanciar el objecto para guardar video del resultado
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29, (640, 480))
    # If que controlar si se ejecuta con informacion de la camara web o de un video
    if Camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("../Include/TestOBj.m4v")
    # Tomar el tiempo para analizar el rendimiento del sistema
    last_time = time.time()
    while True:
        # leer un frame
        ret, frame = cap.read()
        # condicion de terminacion del codigo
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            out.release()
            break
        # Procesar el frame con la red de recontruccion de poses OpenPose
        Poses = OpenPose.pose_construct(frame)
        # Procesar el resultado anterior con la ed Yolo para la deteccion de objetos y poses
        Detections = Yolo.detection(Poses)
        # Pasar la detecciones por el filtro de kalman
        traked_objects=sort.update(Detections)
        # Desglosar la informacion obtenida y acomodarla para su posterior visualizacion
        Final_recongition=[]
        if (len(Detections) > 0):
            for i in range(len(Detections)):
                x1, y1, x2, y2, _, _ = traked_objects[i]
                _, _, _, _, confidences, personId= Detections[i]
                Final_recongition.append([x1, y1, x2, y2, confidences, personId])
        # Funcion que dibuja las detecciones sobre el frame
        result = Yolo.Draw_detection(Final_recongition, Poses)
        # Guardar el frame en el video
        out.write(result)
        # Visualizar el frame en una ventana
        cv2.imshow('window', result)
        # Calcular el tiempo de procesamiento del frame
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

if __name__ == '__main__':
    # Llamar la funcion main
    main(Camera = True)
