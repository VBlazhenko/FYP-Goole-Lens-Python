import os
import sys
import time
import imutils
import webbrowser
import numpy as np
import pandas as pd
from pytesseract import Output
import easyocr
import requests
import argparse
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image
from pytesseract import pytesseract


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'key.json'

def build_model(is_cuda):
    net = cv2.dnn.readNet("yolov5s.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_capture():
    # capture = cv2.VideoCapture("video.mp4") USE VIDEO
    capture = cv2.VideoCapture(0)# USES REAL TIME
    return capture


def load_classes():
    class_list = []
    with open("classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


class_list = load_classes()


def wrap_detection(input_image, output_data):

    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]

        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]

            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]
is_cuda = len(sys.argv) > 1 and sys.argv[1] == "cuda"

net = build_model(is_cuda)
capture = load_capture()

start = time.time_ns()
frame_count = 0
total_frames = 0
fps = -1
label = []

pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# ==================== COLOR DETECTION  ===================

r = g = b = xpos = ypos = 0

index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
df = pd.read_csv('colors.csv', names=index, header=None)

def na_objects():
    na_root = Tk()
    frm = ttk.Frame(na_root, padding=10)
    frm.grid()
    na_root.geometry("240x90")
    na_root.title("UI")
    ttk.Label(frm, text="No objects detected").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=na_root.destroy).grid(column=1, row=0)
    na_root.mainloop()


def getColorName(R, G, B):
    minimum = 10000
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = df.loc[i, 'color_name'] + '   Hex=' + df.loc[i, 'hex']
    return cname


def identify_color(event, x, y, flags, param):
    global b, g, r, xpos, ypos, clicked
    xpos = x
    ypos = y
    b, g, r = frame[y, x]
    b = int(b)
    g = int(g)
    r = int(r)

cv2.namedWindow('image')
cv2.setMouseCallback('image', identify_color)

# ==================== COLOR DETECTION END ===================


# ================== OBJECT DETECTION #######################

while True:
    global prediction_text
    _, frame = capture.read()

    ######## COLOR DETECTION PARAMETERS########

    (grabbed, frame) = capture.read()
    frame = imutils.resize(frame, width=900)
    kernal = np.ones((5, 5), "uint8")

    ###### COLOR DETECTION END PARAMETERS######

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)

    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

    frame_count += 1
    total_frames += 1

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        prediction_text = f"{class_list[classid]}: {confidence:.2f}%"
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(frame, box, color, 2)
        cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        #cv2.putText(frame, prediction_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        # IF WANT TO SEE COLOR DETECTED ON SCREEN - UNCOMMENT BELOW
        # cv2.rectangle(frame, (20, 20), (800, 60), (b, g, r), -1)
        text = getColorName(b, g, r)

        label = class_list[classid]

    if frame_count >= 30:
        end = time.time_ns()
        fps = 1000000000 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()

    with open("data.txt", "a") as f:
        f.write("\n")
        x = str(label)
        f.write(x)

    with open("prediction.txt", "a") as d:
        #print(prediction_text)
        #time.sleep(2)
        d.write("\n")
        #s = str(prediction_text)
        d.write(prediction_text)
        #print(s)

    cv2.imshow('image', frame)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        cv2.imwrite("wrds_det.jpg", frame)
        cv2.imwrite("captured_data_detection.jpg", frame)
        print("")
        print("Colour:", text)

        opening = open("prediction.txt", "r")
        dt = opening.read()
        dt_into_lst = dt.replace("\n", " ").split(".")
        # print(dt_into_lst)

        smth_list = []
        for t in dt_into_lst:
            smth_list.extend(t.split(" "))

        nw_list = []
        for list_obj in smth_list:
            if list_obj not in nw_list:
                nw_list.append(list_obj)
        del nw_list[0]
        print("Objects with their %: ", nw_list)

        ####################### CONVERTING LIST INTO A DICTIONARY ##############################

        d = {}
        for x in nw_list:
            if x.endswith(':'):
                k = x[:-1]
            else:
                val = int(x.rstrip('%'))
                d[k] = max(d.get(k, val), val)

        out = max(d, key=d.get)
        result = [k for k, v in d.items() if v >= 65]

        print("Objects with a percentage higher than 65%: ", result)
        if len(result) == 0:
            na_objects()
            exit()


        ####################### CONVERTING LIST INTO A DICTIONARY END ##############################

        opening.close()

        with open("prediction.txt", "w") as b:
            b.write("")

        my_file = open("data.txt", "r")
        data = my_file.read()
        data_into_lst = data.replace("\n", " ").split(".")

        sth_lst = []
        for i in data_into_lst:
            sth_lst.extend(i.split(" "))

        new_lst = []
        for lst_object in sth_lst:
            if lst_object not in new_lst:
                new_lst.append(lst_object)
        del new_lst[0]

        print("All Objects Detected", new_lst)
        length_list = len(new_lst)
        #print("Number of objects in total: ", length_list)
        my_file.close()

        with open("data.txt", "w") as x:
            x.write("")
        break

# ================== OBJECT DETECTION END #######################

#==============TEXT/LOGO DETECTION + OFFLINE MODE==============

capture = cv2.VideoCapture(0)

def which_button(button_press):
    if button_press == "Logo":
        take_logo()
        #print("Logo")
    elif button_press == "Text":
        text_detection_offline() # NOT USING GOOGLE API
        #print("Text")

def window():
    # Create an instance of Tkinter frame
    win = Tk()
    win.title("Logo/ Text")
    # Set the geometry of Tkinter frame
    win.geometry("150x150")

    # Create Buttons with proper position
    button1 = ttk.Button(win, text="Logo", command=lambda x="Logo": which_button(x))
    button1.pack(side=TOP)
    button2 = ttk.Button(win, text="Text", command=lambda x="Text": which_button(x))
    button2.pack(side=TOP)

    win.mainloop()



def take_logo():
    global extra

    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # Loads the image into memory
    with io.open("captured_data_detection.jpg", 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    # Performs label detection on the image file
    response = client.logo_detection(image=image)
    annotations = response.logo_annotations
    for annotation in annotations:
        extra = annotation.description
        print(extra)
    button_window(extra)
    return extra


def text_detection_offline():
    global extra
    reader = easyocr.Reader(['ch_sim','en'], gpu=False)
    result = reader.readtext('captured_data_detection.jpg', detail = 0)
    extra = ' '.join(result)
    print(extra)
    button_window(extra)
    return extra


def web_site_online(url='http://www.google.com/', timeout=5):
    try:
        req = requests.head(url, timeout=timeout)
        # HTTP errors are not raised by default, this statement does that
        req.raise_for_status()
        window()
    except requests.HTTPError as e:
        print("Checking internet connection failed, status code {0}.".format(
        e.response.status_code))
    except requests.ConnectionError:
        text_detection_offline()

#============END TEXT/LOGO DETECTION + OFFLINE MODE============

# ================== TKINTER #######################

def button_window(argument, url='http://www.google.com/', timeout=5):
    window = Tk()
    window.geometry("285x250")

    # Create a LabelFrame
    labelframe = LabelFrame(window)
    window.title("Detected Objects")

    # Define a canvas in the window
    canvas = Canvas(labelframe)
    canvas.pack(side=RIGHT, fill=BOTH, expand=1)
    labelframe.pack(fill=BOTH, expand=1, padx=30, pady=30)

    start = 0
    end = length_list
    step = 1

    for y in range(start, end, step):
        x = y
        sec = new_lst[x:x+step]

        # URL SECTION
        init_url = "amazon.co.uk/s?k="
        x = str(sec)

        final_url = init_url + x
        cutting = final_url.replace("'", '')
        cutting_two = cutting.replace("[", "")
        cutting_three = cutting_two.replace("]", "")

    init_url = "https://amazon.co.uk/s?k="

    to_string = str(text)
    # removing hex
    remove_hx = text.replace("Hex=#", "")
    # breaking into separate words
    brk_lst = remove_hx.split()
    # removing last character
    rm_lst_char = brk_lst[:-1]
    # pulling together words
    over_res = " ".join(str(x) for x in rm_lst_char)

    # print("URL consists of the following:")
    # print("Initial URL-", init_url)
    # print("Detected objects-", result)
    # print("Detected colour-", over_res)

    def open(url):
        webbrowser.open(url)

    amz_url = "https://amazon.co.uk/s?k="
    ebay_url = "https://www.ebay.co.uk/sch/i.html?_nkw="

    def close_function():
        sys.exit()

    try:
        req = requests.head(url, timeout=timeout)
        req.raise_for_status()
        Label(canvas, text="Detected Objects").grid(row=0, column=1)
        Label(canvas, text="Amazon").grid(row=1, column=0)
        Label(canvas, text="Ebay").grid(row=1, column=3)

        x, y = 2, 0
        for item in result:
            final_url = amz_url + item + " " + over_res + " " + extra
            #Button(canvas, text=item, command=lambda url=final_url: open(url)).grid(row=x, column=y)
            btn1 = Button(canvas, text=item, command=lambda  url=final_url: open(url))
            if result.index(item) == 7:
                x = 0
                y = y+1
            btn1.grid(row=x, column=y)
            x = x + 1

        a,b = 2, 3
        for item in result:
            final_url = ebay_url + item + " " + over_res + " " + extra
            #Button(canvas, text=item, command=lambda url=final_url: open(url)).grid(row=2, column=2)
            btn2 = Button(canvas, text=item, command=lambda url=final_url: open(url))
            if result.index(item) == 7:
                a = 0
                b = b + 1
            btn2.grid(row=a, column=b)
            a = a + 1
        Button(canvas, text="Close", command=close_function).grid(row=70, column=1)

    except requests.HTTPError as e:
        print("Checking internet connection failed, status code {0}.".format(
        e.response.status_code))

    except requests.ConnectionError:
        # NOT CONNECTED
        for item in result:
            final_url = amz_url + item + " " + over_res + " " + extra
            Label(canvas, text="You are not connected to the internet").pack()
            Label(canvas, text="The URL you requested is: ").pack()
            Label(canvas, text=final_url).pack()

    window.mainloop()

# ================== TKINTER END #######################

# ================== RESULTS #######################

web_site_online()
#button_window(extra)
capture.release()
