# Importar librerias
import cv2
import numpy as np

class Pose_Detection:
    # Inicializacion de la clase
    def __init__(self, MODE="MPI", use_gpu=True):
        # Se selecciona que modelo de recontruccion de poses utilizar
        if MODE is "COCO":
            # Se carga la arquitectura y los pesos de la red
            self.protoFile = "../Include/coco/pose_deploy_linevec.prototxt"
            self.weightsFile = "../Include/coco/pose_iter_440000.caffemodel"
            # Se especifica la cantidad puntos que puede predecir la red
            self.nPoints = 18
            # Se crea una lista con la informacion de interconexion de los puntos
            self.POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                          [11, 12],
                          [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

        elif MODE is "MPI":
            # Se carga la arquitectura y los pesos de la red
            self.protoFile = "../Include/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
            self.weightsFile = "../Include/mpi/pose_iter_160000.caffemodel"
            # Se especifica la cantidad puntos que puede predecir la red
            self.nPoints = 15
            # Se crea una lista con la informacion de interconexion de los puntos
            self.POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10],
                          [14, 11],
                          [11, 12], [12, 13]]
            # Se declara los colores que tendra cada punto seguin su ubicacion
            self.Colors = [(255, 0, 0),  # 1. Cabeza
                      (0, 255, 0),  # 2. Cuello
                      (0, 0, 255),  # 3. Hombro derecho
                      (255, 255, 0),  # 4. Codo derecho
                      (0, 255, 255),  # 5. muñeca derecha
                      (255, 0, 255),  # 6. Hombro Izquierdo
                      (133, 10, 10),  # 7. Codo Izquierdo
                      (10, 133, 10),  # 8. Muñeca izquierda
                      (10, 10, 133),  # 9. Cadera derecha
                      (133, 133, 10),  # 10. Rodilla derecha
                      (133, 10, 133),  # 11. Tobillo derecho
                      (10, 133, 133),  # 12. Cadera Izquierda
                      (133, 133, 133),  # 13. Rodilla Izquierda
                      (50, 50, 50),  # 14. Tobillo Izquierdo
                      (50, 100, 50)]  # 15. Ombligo
        # Se define el tamaño de entrada de la imagen
        self.inWidth = 368
        self.inHeight = 368
        self.threshold = 0.2
        # Se carga la arquitectura y pesos de la red con OpenCV
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        # Se carga el modelo en GPU en caso de estar seleccionada
        if use_gpu==True:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Funcion de reconstruccion de poses
    def pose_construct(self, frame):
        frameCopy = np.copy(frame)
        # Se obtiene el tamaño de la imagen original
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        # Se escala la imagen al tamaño de entrada a red y se normaliza
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (self.inWidth, self.inHeight), (0, 0, 0), swapRB=False,
                                        crop=False)
        # Se pasa la imagen escalada a la red
        self.net.setInput(inpBlob)
        # Se realiza una prediccion
        output = self.net.forward()
        # Se obtiene el tamaño de la salida
        H = output.shape[2]
        W = output.shape[3]
        points = []
        # Se recorre los puntos predecidos
        for i in range(self.nPoints):
            # Se desglosa la informacion de la prediccion
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            # Se calcula la ubicacion del punto predicho en la imagen original
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H
            # Se verifica que la probabilidad del que el punto es correcto supere el threshold seleccionado
            if prob > self.threshold:
                # Se dibuja el punto predicho y se guarda en una lista
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
                points.append((int(x), int(y)))
            else:
                points.append(None)
        # Se recorre la lista que contiene la informacion de la intercionexion de los puntos
        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]
            # Se verifica que ambos puntos existan
            if points[partA] and points[partB]:
                # Se verifica que esten dentro de los puntos de interes
                if partA < 8 or partA == 15:
                    # Se dibuja la linea entre los puntos en la imagen
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
                    # Se dibuja el primer punto en la imagen
                    cv2.circle(frame, points[partA], 3, self.Colors[partA], thickness=-1, lineType=cv2.FILLED)
                    # Se dibuja el segundo punto en la imagen
                    cv2.circle(frame, points[partB], 3, self.Colors[partB], thickness=-1, lineType=cv2.FILLED)
        # Se retorna la imagen
        return frame