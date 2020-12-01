
from collections import deque
import pickle
import cv2
import numpy as np
import time
import ast
from utils import *
import tensorflow_hub as hub
import concurrent.futures
from tensorflow.keras import layers
import tensorflow as tf



# Load Yolo
net = cv2.dnn.readNet("./data/yolov4-tiny.weights", "./data/yolov4-tiny.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture('vid_short.mp4')


mouse_pts = []

model = tf.keras.models.load_model('./model/resnet191020.h5')
model.summary()

#lb = pickle.loads(open(args["label"], "rb").read())
#lb = ["football","tennis","weight_lifting"]

lb = ['Fire', 'Normal Car', 'Normal', 'Road Accident', 'Shooting', 'Violence']
#model.summary()
# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.,
                                   )





my_file = open("./test.txt","a+")
def get_mouse_points(event, x, y, flags, param):
    # Used to mark 4 points on the frame zero of the video that will be warped
    # Used to mark 2 points on the frame zero of the video that are 6 feet away
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        file1=open("./test.txt","a")
        cv2.circle(image, (x, y), 10, (0, 255, 255), 10)
        if "mouse_pts" not in globals():
            mouse_pts = []
        if(len(mouse_pts)==6):
            file1.write(str(mouse_pts))
        file1.close()
        mouse_pts.append((x, y))
        print("Point detected")
        print(mouse_pts)


def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False



scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)
frame_num = 0
total_pedestrians_detected = 0
total_six_feet_violations = 0
total_pairs = 0
abs_six_feet_violations = 0
pedestrian_per_sec = 0
sh_index = 1
sc_index = 1

cv2.namedWindow("image")
cv2.setMouseCallback("image", get_mouse_points)
num_mouse_points = 0
first_frame_display = True

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    if frame_id == 1:
    # Ask user to mark parallel points and two points 6 feet apart. Order bl, br, tr, tl, p1, p2
        while True:
            image = frame
            file = open('./test.txt','r')
            s = file.read()
            if s:
                x = ast.literal_eval(s)
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if s:
                if len(mouse_pts) == 7 or len(x) == 6:
                    cv2.destroyWindow("image")
                    mouse_pts = x
                    break
            first_frame_display = False
        four_points = mouse_pts

        M = perspective(frame, four_points[0:4])
        pts = src = np.float32(np.array([four_points[4:]]))
        warped_pt = cv2.perspectiveTransform(pts, M)[0]
        d_thresh = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        bird_image = np.zeros(
            (int(height * scale_h), int(width * scale_w), 3), np.uint8
        )

        bird_image[:] = SOLID_BACK_COLOR
        pedestrian_detect = frame

   # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                h = int(detection[3] * height)
                w = int(detection[2] * width)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        flat_box = indexes.flatten()
        pairs = []
        center = []
        status = []
        for i in flat_box:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)

        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j])

                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
                    
        index = 0
        for i in flat_box:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()

    pedestrian_boxes, num_pedestrians = indexes, len(indexes)

#   if len(indexes) > 0:
#       pedestrian_detect = bird_eye_view_plot(frames, boxes, M, scale_w, scale_h)

    canvas = np.zeros((200,200,3))
    canvas[:] = (0,0,0)
    text = "people:{}".format(len(pedestrian_boxes))
    cv2.putText(canvas, text, (35,50), cv2.FONT_HERSHEY_SIMPLEX,
           1.0, (0,255,0), 5)
    cv2.imshow('info',canvas)


    # make predictions on the frame and then update the predictions
    # queue
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame = train_datagen.standardize(frame)   

    
    preds = model.predict(np.expand_dims(frame, axis=0),workers=6,use_multiprocessing=True)[0]
    Q.append(preds)
     

    for (i,(lab, prob)) in enumerate(zip(lb, preds)):
        text= "{}:{:.2f}%".format(lab, prob*100)
        w = int(prob*300)
        cv2.rectangle(canvas, (7, (i*35) +5), 
            (w, (i*35)+35), (0,0,255), -1)
        cv2.putText(canvas, text, (10,(i*35)+23), cv2.FONT_HERSHEY_SIMPLEX,0.45, (255,255,255),2)

    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb[i]
    print(label)
    # draw the activity on the output frame
    text = "{}".format(label)
    cv2.putText(output, text, (105, 50), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 5)
    
    cv2.imshow("probs", canvas)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(output, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    cv2.imshow("Image", output)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 2, color, 1)

    if len(pedestrian_boxes) > 0:
        warped_pts, bird_image = display_points(
            frame, boxes
        )


    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
    