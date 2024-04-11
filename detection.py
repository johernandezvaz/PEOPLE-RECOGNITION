import cv2
import numpy as np

# Cargar la red neuronal YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Cargar las clases
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Cargar la imagen
img = cv2.imread("kid-games.webp")
height, width, _ = img.shape

# Preprocesar la imagen
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Obtener las salidas de la red neuronal
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

# Listas para almacenar las coordenadas, confianzas y clases de los objetos detectados
boxes = []
confidences = []
class_ids = []

# Inicializar la lista de colores
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Recorrer las salidas de la red neuronal
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Umbral de confianza
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar supresión de no máximos para eliminar detecciones superpuestas
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Inicializar el contador de personas
person_count = 0

# Verificar las detecciones de personas
font = cv2.FONT_HERSHEY_PLAIN
if len(indexes) > 0:
    for i in indexes.flatten():
        label = str(classes[class_ids[i]])
        if label == 'person':
            person_count += 1
            if i < len(colors):
                color = colors[i]
            else:
                color = (0, 255, 0)  # Color verde para clases adicionales
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]), color, 2)
            cv2.putText(img, label, (boxes[i][0], boxes[i][1] + 30), font, 3, color, 3)


# Mostrar el resultado
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Imprimir el conteo de personas detectadas
print("Número de personas detectadas:", person_count)
