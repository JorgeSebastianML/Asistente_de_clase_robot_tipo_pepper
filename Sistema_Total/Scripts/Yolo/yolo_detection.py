# Importar librerias
import cv2
import numpy as np
import copy

class Yolo_Detection:
    # Inicializacion de la clase
    def __init__(self, use_gpu, confidence=0.6, threshold=0.5, size=608):
        # Ruta en donde se almacena la arquitectura de la red
        self.cfg_path = "../Include/yolo-obj-test.cfg"
        # Ruta en donde se almacena los pesos de la red
        self.weights_path = "../Include/yolo-obj-test_best.weights"
        # Rita en donde se almacena la lista de nombres correspondientes a la clases que la red predice
        self.class_names_path = "../Include/obj.names"
        # Se define la confianza el treshold para la deteccion
        self.confidence = confidence
        self.threshold = threshold
        # Se define el tamaño de la imagen para entrar a la red
        self.size_imge = size
        # Se carga la red junto con sus pesos por medio de la libreria OpenCV
        self.model = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        # Se habilita el uso de la GPU
        if use_gpu == True:
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # Se identifica las capas de la red, para identificar las neuronas de salida
        self.output_layer_names = self.model.getLayerNames()
        self.output_layer_names = [self.output_layer_names[i[0] - 1] for i in self.model.getUnconnectedOutLayers()]
        # Se guarda en una lista los nombres de las clases
        self.labels = open(self.class_names_path).read().strip().split("\n")

    # Funcion encardada de la deteccion
    def detection(self, image):
        # Se determina el tamaño original de las imagenes de entrada
        (W, H) = (None, None)
        if W is None or H is None:
            (H, W) = image.shape[:2]
        # Se escala la imagen al tamaño de entrada a red y se normaliza
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size_imge, self.size_imge), swapRB=True, crop=False)
        # Se pasa la imagen escalada a la red
        self.model.setInput(blob)
        # Se realiza una prediccion
        model_output = self.model.forward(self.output_layer_names)
        # Se revisa la prediccion
        boxes = []
        confidences = []
        class_ids = []
        for output in model_output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                # Se verifica que la prediccion tenga un confianza mayor a la umbral seleccionada
                if confidence > self.confidence:
                    # Se guarda la informacion de las detecciones con suficiente confianza
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    print(confidence)
        # Se organiza la informacion obtenida de la deteccion
        outputs_wrapper = list()
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                outputs_wrapper.append([x, y, w + x, h + y, confidences[i], class_ids[i]])
        # se retorna toda la informacion correspondiente a la prediccion
        return np.array(outputs_wrapper)

    # Funcion para dibujar la deteccion en frame
    def Draw_detection(self, yolo_output, image):
        image_ = copy.deepcopy(image)
        # Desglosar la informacion de la deteccion
        for bbox in yolo_output:
            color = (255, 0, 0)
            # Crear bounding box
            bbox = list(np.array(bbox).astype(int))
            x, y, x2, y2, object_id, Name = bbox
            # Guardar el nombre de la prediccion
            Name = self.labels[Name]
            print(Name)
            # Dibujar el bounding box en el frame
            cv2.rectangle(image_, (x, y), (x2, y2), color, 2)
            # Escribir la prediccion en el frame
            cv2.putText(image_, Name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        # Se retorna la imagen
        return image_
