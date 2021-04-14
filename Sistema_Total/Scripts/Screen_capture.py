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
        Poses = OpenPose.pose_construct(img)
        Detections  = Yolo.detection(Poses)
        traked_objects=sort.update(Detections)

        Final_recongition=[]
        if (len(Detections) > 0):
            for i in range(len(Detections)):
                x1, y1, x2, y2, _, _ = traked_objects[i]
                _, _, _, _, confidences, personId= Detections[i]
                Final_recongition.append([x1, y1, x2, y2, confidences, personId])
        result = Yolo.Draw_detection(Final_recongition, Poses)
        out.write(result)
        cv2.imshow('window', result)
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            out.release()
            break

if __name__ == '__main__':
    # Calling main() function
    main()
