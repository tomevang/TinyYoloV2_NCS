from mvnc import mvncapi as mvnc
import math
import cv2
import numpy as np
import tensorflow as tf
from numpy import array
import time
import sys

IMAGE = "./images/000002.jpg"
GRAPH_PATH = "./graph"


def sigmoid(p):
    return 1.0 / (1 + math.exp(-p * 1.0))


def overlap(x1, w1, x2, w2):  # x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left


def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area


def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes, key=lambda d: d[7])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue

        truth = sorted_boxes[i]
        for j in range(i + 1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1

    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res


def post_processing(output):

    res = output.astype(np.float32)
    res=np.reshape(res,(13,13,125))

    swap = np.zeros((13 * 13, 5, 25))

    index = 0
    for h in range(13):
        for w in range(13):
            for c in range(125):
                i=h*13 + w
                j = int(c/25)
                k = c%25
                swap[i][j][k]=res[h][w][c]

    biases = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    boxes = list()
    for h in range(13):
        for w in range(13):
            for n in range(5):
                box = list();
                cls = list();
                s = 0;
                x = (w + sigmoid(swap[h * 13 + w][n][0])) / 13.0;
                y = (h + sigmoid(swap[h * 13 + w][n][1])) / 13.0;
                ww = (math.exp(swap[h * 13 + w][n][2]) * biases[2 * n]) / 13.0;
                hh = (math.exp(swap[h * 13 + w][n][3]) * biases[2 * n + 1]) / 13.0;
                obj_score = sigmoid(swap[h * 13 + w][n][4]);
                for p in range(20):
                    cls.append(swap[h * 13 + w][n][5 + p]);

                large = max(cls);
                for i in range(len(cls)):
                    cls[i] = math.exp(cls[i] - large);

                s = sum(cls);
                for i in range(len(cls)):
                    cls[i] = cls[i] * 1.0 / s;

                box.append(x);
                box.append(y);
                box.append(ww);
                box.append(hh);
                box.append(cls.index(max(cls)) + 1)
                box.append(obj_score);
                box.append(max(cls));
                box.append(obj_score * max(cls))
                # print("these are the values of box 5 and 6", box[5], box[6])
                # if score
                if box[5] * box[6] > 0.1:
                    boxes.append(box);
    res = apply_nms(boxes, 0.35)
    label_name = {0: "bg", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle", 6: "bus", 7: "car",
                  8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog", 13: "horse", 14: "motorbike",
                  15: "person", 16: "pottedplant", 17: "sheep", 18: "sofa", 19: "train", 20: "tvmonitor"}
    w = img_cv.shape[1]
    h = img_cv.shape[0]

    for box in res:

        xmin = (box[0] - box[2] / 2.0) * w;
        xmax = (box[0] + box[2] / 2.0) * w;
        ymin = (box[1] - box[3] / 2.0) * h;
        ymax = (box[1] + box[3] / 2.0) * h;
        if xmin < 0:
            xmin = 0
        if xmax > w:
            xmax = w
        if ymin < 0:
            ymin = 0
        if ymax > h:
            ymax = h

        cv2.rectangle(img_cv,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
        print (label_name[box[4]],xmin, ymin, xmax, ymax)
        
        label_text = label_name[box[4]] + " " + str("{0:.2f}".format(box[5]*box[6]))
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text
        
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_left = int(xmin)
        label_top = int(ymin) - label_size[1]
        label_right = label_left + label_size[0]
        label_bottom = label_top + label_size[1]


        cv2.rectangle(img_cv, (label_left-1, label_top-5),(label_right+1, label_bottom+1), label_background_color, -1)
        cv2.putText(img_cv, label_text, (int(xmin), int(ymin-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imshow('YOLO detection',img_cv)
    cv2.waitKey(10000)

# Entry point for program

img = cv2.imread(IMAGE)
img_cv = img
img = np.divide(img, 255.0) 
img = cv2.resize(img, (416, 416), cv2.INTER_LINEAR)
img = img[:,:,::-1]
img = img.astype(np.float32)

mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)

devices = mvnc.enumerate_devices()
if len(devices) == 0:
    print('No devices found')
    quit()

device = mvnc.Device(devices[0])

device.open()

#Load blob
with open(GRAPH_PATH, mode='rb') as f:
    graph_in_memory = f.read()

graph = mvnc.Graph(GRAPH_PATH)

fifo_in, fifo_out = graph.allocate_with_fifos( device, graph_in_memory )


graph.queue_inference_with_fifo_elem(fifo_in, fifo_out, img, 'user object')
output, userobj = fifo_out.read_elem()

post_processing(output)

fifo_in.destroy()
fifo_out.destroy()
graph.destroy()
device.close()
device.destroy()


