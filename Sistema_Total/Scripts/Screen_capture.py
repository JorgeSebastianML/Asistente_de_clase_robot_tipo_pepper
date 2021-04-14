# Importar librerias
import pyscreenshot as ImageGrab
import numpy as np
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
def main():
    # Instanciar objeto Yolo, para realizar el reconocimiento de objetos y de poses
    Yolo = Yolo_Detection(use_gpu=True, confidence=0.45, threshold=0.3, size=608)
    # Instanciar objeto OpenPose para la recontrucion de poses
    OpenPose = Pose_Detection()
    # Instanciar objeto sort para el uso del filtro de kalman
    sort = Sort()
    # Instanciar el objecto para guardar video del resultado
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
    # Tomar el tiempo para analizar el rendimiento del sistema
    last_time = time.time()
    while True:
        # leer un frame tomado de la pantalla
        screen = np.array(ImageGrab.grab(bbox=(67, 57, 707, 537)))
        # Realizar una conversion de color de RGB a BGR
        img= cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        # Procesar el frame con la red de recontruccion de poses OpenPose
        Poses = OpenPose.pose_construct(img)
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
        # condicion de terminacion del codigo
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            out.release()
            break

if __name__ == '__main__':
    # Llamar la funcion main
    main()
