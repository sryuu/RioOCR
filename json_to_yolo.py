import json
import glob
import cv2
import csv
import pprint
import os
from tqdm import tqdm

json_open = open("digitStruct" + ".json", 'r')
json_load = json.load(json_open)
#box = json_load["boxes"]
#filename = json_load["filename"]

count = 0
with open("./train.txt", mode='w') as f1:
    with open("./test.txt", mode='w') as f2:
        for data in tqdm(json_load):
            filen = data["filename"].replace("png","")
            if os.path.isfile("D:/DATASET/train/" + data["filename"]) is True:
                im = cv2.imread("D:/DATASET/train/" + data["filename"])
                h, w, c = im.shape
                with open("./yolo/"+filen+"txt", mode='w') as f3:
                    if count == 10:
                        count = 0
                        f2.write("data/obj/" + data["filename"]+"\n")
                    else:f1.write("data/obj/" + data["filename"]+"\n")
                    bbox = data["boxes"]
                    for ins in bbox:
                        list_of_label = []
                        for un in range(10):
                            if int(ins["label"]) == un:
                                labelnumber = un + 1
                                list_of_label.append(str(labelnumber))
                                break
                            elif int(ins["label"]) == 10:
                                labelnumber = 1
                                list_of_label.append(str(labelnumber))
                                break
                        list_of_label.append(str((ins["left"]+(ins["width"]/2))/w))
                        list_of_label.append(str((ins["top"]+(ins["height"]/2))/h))
                        list_of_label.append(str(ins["width"]/w))
                        list_of_label.append(str(ins["height"]/h))
                        for txw in range(4):
                            f3.write(list_of_label[txw] +" ")
                        f3.write(list_of_label[4])
                        f3.write("\n")
                count += 1