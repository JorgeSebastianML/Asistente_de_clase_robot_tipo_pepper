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
    Yolo = Yolo_Detection(use_gpu=True, confidence=0.45, threshold=0.3, size=608)
    OpenPose = Pose_Detection()
    sort = Sort()
    last_time = time.time()
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 480))
    while True:
        screen = np.array(ImageGrab.grab(bbox=(67, 57, 707, 537)))
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
