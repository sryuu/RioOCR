import numpy as np
import cv2
import os
from keras.layers import Input
from yolo4.model import yolo_eval, yolo4_body
from decode_np import Decode
from operator import itemgetter

#1.landmarkの前準備
ns=[]
with open("./model_data/my_classes.txt",encoding='utf-8') as f:
    s = f.readlines()
    for ss in s:
        ns.append(ss[0])
    print(ns)

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path,encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

model_path = 'yolo4_weight.h5'
anchors_path = 'model_data/yolo4_anchors.txt'
classes_path = 'model_data/my_classes.txt'

class_names = get_class(classes_path)
anchors = get_anchors(anchors_path)

num_anchors = len(anchors)
num_classes = len(class_names)

model_image_size = (128, 128)
#しきい値
conf_thresh = 0.2
nms_thresh = 0.45

yolo4_model = yolo4_body(Input(shape=model_image_size+(3,)), num_anchors//3, num_classes)

model_path = os.path.expanduser(model_path)
assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

yolo4_model.load_weights(model_path)

_decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

def myOCR(image,bun):
    # 4.ＯＣＲ実行
    sorty = []
    sortx = []
    sortclass = []
    li1=[]
    pcount=0
    sco=[]
    cla=[]
    height, width, channels = image.shape[:3]
    changed_image, boxes, scores, classes = _decode.detect_image(image, True)
    if classes is None:
        print("認識可能な文字がありません")
        pass
    else:
        for res in range(len(classes)):
            sco.append(scores[res])
            cla.append(class_names[classes[res]])
        resofyolo = dict(zip(cla,sco))
        pprint.pprint(resofyolo)
        changed_image = cv2.resize(changed_image, (300,300))
        for long in range(len(classes)):
            sorty.append(int(boxes[long][1]))
            sortx.append(int(boxes[long][0]))
            sortclass.append(ns[int(classes[long])])
        li1=[sortclass,sortx,sorty]
        li1 = np.array(li1).T
        li1 = li1.tolist()
        for toint in li1:
            toint[1] = int(toint[1])
            toint[2] = int(toint[2])
        sorted_data = sorted(li1, key=itemgetter(1,2))
        #print(sorted_data)
        #print(len(sorted_data))
        #print("文字だけソートして抽出:\n" + str(printer))
        #ここで文字列ごとに分ける
        #ocr行しきい値
        printer1 = np.zeros(bun)
        l_n_str = [str(n) for n in printer1]
        for mojiretu in range(bun):
            for moji in sorted_data:
                if mojiretu*height/bun <= moji[2] < (mojiretu+1)*height/bun:
                    if pcount == 0:
                        l_n_str[mojiretu] = moji[0]
                    else:l_n_str[mojiretu] += moji[0]
                    pcount += 1
            pcount=0
        print(cat+":")
        for ocr in l_n_str:
            if ocr == "0.0":
                continue
            print(ocr)
    cv2.imshow('image', changed_image)
    cv2.waitKey(0)

#makedata = 0
if __name__ == '__main__':
    im = cv2.imread("path of image")#input image data here
    myOCR(im,"画像を分割する割合")
