# Importar librerias
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

# Main
def main(Camera = False):
    # Instanciar objeto Yolo, para realizar el reconocimiento de objetos y de poses
    Yolo = Yolo_Detection(use_gpu=True, confidence=0.45, threshold=0.3, size=608)
    # Instanciar objeto OpenPose para la recontrucion de poses
    OpenPose = Pose_Detection()
    # Instanciar objeto sort para el uso del filtro de kalman
    sort = Sort()
    last_time = time.time()
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29, (640, 480))
    if Camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("../Include/TestOBj.m4v")

    while True:
        ret, frame = cap.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            out.release()
            break
        #img= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        Poses = OpenPose.pose_construct(frame)
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

if __name__ == '__main__':
    # Calling main() function
    main(Camera = True)
