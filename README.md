# Sistema Multimodal para un asistente robótico por medio de aprendizaje por refuerzo

Este sistema multimodal fue construido por medio de aprendizaje por refuerzo, que le permitirá al robot tipo Pepper ser un apoyo para los profesores de laboratorio, en donde por medio de sus sensores infrarrojos, sus cámaras 2D y 3D obtiene información de su entorno. Que posteriormente son procesadas por múltiples modelos de aprendizaje automático, para obtener así una mayor información del ambiente, como una detección y reconocimiento de los objetos que el robot ve y la capacidad de reconstruir y clasificar las poses de los estudiantes.

<img src="https://github.com/JorgeSebastianML/Asistente_de_clase_robot_tipo_pepper/blob/main/Img/Arquitectura_Español.png " width="300" height="500">

## Reconstrucción de poses 

Se implemento la red OpenPose encontrada en el repositorio https://github.com/CMU-Perceptual-Computing-Lab/openpose por medio de OpenCV versión 4.x. Esta red fue modificada para tener en cuenta solo la recontrucion del torso para arriba. 

## Clasificación de poses

Se implemento la red convolucional Yolo v4 encontrada en el repositorio https://github.com/AlexeyAB/darknet por medio de OpenCV versión 4.x. la cual fue entrenada con las poses de interes Pregunta, Alto, Ninguna. Esta red tuvo un desempeño en datos de validación del 89.84% de accurracy. 

## Clasificación de objetos

Se implemento la red convolucional Yolo v4 encontrada en el repositorio https://github.com/AlexeyAB/darknet por medio de OpenCV versión 4.x. la cual fue entrenada con los objetos de interes, Botellas, Snacks, Osciloscopio, Multímetro, Generador de señales, Aparatos electrónicos, Fuente de alimentación. Esta red tuvo un desempeño en datos de validación del 80.8% de mean Average Precision. 

## Aprendizaje por refuerzo

Se implemento la arquitectura Proximal Police Optimization y por medio del simulador P.E.R. Lab encontrado en el repositorio https://github.com/JorgeSebastianML/Unity_simulation_pepper_robot, se realizo una busqueda de grilla para encontrar la mejor arquitectura de red para este problema y la mejor definición de recompensas. 
