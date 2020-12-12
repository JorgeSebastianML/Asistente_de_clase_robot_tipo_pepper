import numpy as np
import cv2
import time
from Yolo.yolo_detection import Yolo_Detection
from Open_pose.pose_detection import Pose_Detection
from Kalman_filter.sort import Sort
import warnings
warnings.filterwarnings("ignore")

def main():
    Yolo = Yolo_Detection(use_gpu=True, confidence=0.45, threshold=0.3, size=608)
    OpenPose = Pose_Detection()
    sort = Sort()
    last_time = time.time()
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 29, (1920, 1080))
    cap = cv2.VideoCapture("../Include/Robot_Vision.mp4")
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
    main()
