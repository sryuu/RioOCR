from keras.preprocessing.image import load_img
from keras.models import load_model
import glob
import numpy as np
from keras.utils import np_utils
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import helpers
import cv2
import os
import csv
import pprint
import matplotlib.pyplot as plt
import os
import pyocr
import pyocr.builders
from keras import backend as K
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

class_to_category_number = {
    'floordisplay':14,
    'stairname':15,
    'nameplate':16,
}

labell=[]
with open('label.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        labell.append(row)

num_of_cat = []
coc = []
for te in labell[1:]:
    if not te[13]:
        break
    num_of_cat.append(te[12])
    coc.append(te[13])
len_of_cat = len(num_of_cat)

num_of_ins = []
coi = []
for te in labell[1:]:
    if not te[15]:
        break
    num_of_ins.append(te[14])
    coi.append(te[15])
len_of_ins = len(num_of_ins)

dic_of_cat = dict(zip(num_of_cat,coc))
dic_of_ins = dict(zip(num_of_ins,coi))

path = './test2/*.jpg'
data = './test/360pic0_1.jpg'
flist = glob.glob(path)

def myOCR(image):
    # 4.ＯＣＲ実行
    sorty = []
    sortx = []
    sortclass = []
    li1=[]
    pcount=0
    printer1=[0,0]
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
        bun = 3
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
    model = load_model('./models/landmark2.h5')
    with open('test.txt', 'w') as f:
        a = 1
        for i in flist:
            #画像の読み込み
            im = cv2.imread(str(i))
            jy = 480
            jx = jy*2
            size = (jx,jy)
            x = cv2.resize(im, size)
            x = np.float32(x) / 255.0
            x = np.array(x)
            x = x[np.newaxis]

            # 読み込んだ学習済みモデルで予測
            y = model.predict(x)
            """
            print("space_category")
            print(y[0].shape)
            print(np.argmax(y[0][0]))
            
            print("instance_category")
            print(y[2].shape)
            print(np.argmax(y[2][0]))
            """
            for k,v in dic_of_cat.items():
                if np.argmax(y[0][0]) == int(k):
                    cc = v
                    break
            for k2,v2 in dic_of_ins.items():
                if np.argmax(y[2][0]) == int(k2):
                    cc2 = v2
                    break
            print(str(i))
            a = a + 1
            print("問"+str(a-1)+"-------撮影場所:" + cc2 + cc + "-------"+"\n")

            if np.argmax(y[0][0]) == 0:
                yp2 = y[2][0][0:2]
                coi = coi[0:2]
            elif np.argmax(y[0][0]) == 1:
                yp2 = y[2][0][2:6]
                coi = coi[2:6]
            elif np.argmax(y[0][0]) == 2:
                yp2 = y[2][0][6:8]
                coi = coi[6:8]
            elif np.argmax(y[0][0]) == 3:
                yp2 = y[2][0][8:17]
                coi = coi[8:17]
            elif np.argmax(y[0][0]) == 4:
                yp2 = y[2][0][17:21]
                coi = coi[17:21]
            elif np.argmax(y[0][0]) == 5:
                yp2 = y[2][0][21:23]
                coi = coi[21:23]
            print(yp2)
            
            fig = plt.figure()
            xp = range(len_of_cat)
            yp = y[0][0]
            ax1 = fig.add_subplot(2,1,1)
            ax1.bar(xp, yp, tick_label = coc) 
            ax1.set_title("space_Probability_distribution")

            xp2 = range(len(yp2))
            yp2 = y[2][0]
            ax2 = fig.add_subplot(2,1,2)
            ax2.bar(xp2, yp2, tick_label = coi) 
            ax2.set_title("instance_Probability_distribution")

            plt.show()
            
            #semantic
            #print((y[1][0].shape))
            #print(np.argmax(y[1][0][0][0]))
            sempic = y[1][0]
            #print("semantic")
            #print(sempic.shape)
            sempic = np.argmax(sempic,axis = -1)
            print(sempic.shape)
            h,w = sempic.shape
            #print(sempic[0][0])

            #ランドマークデータを分別
            if np.argmax(y[0][0]) == 0 or np.argmax(y[0][0])== 3:
                #ランドマークの始点と終点を検出
                for cat,num in class_to_category_number.items():
                    if num in sempic:
                        newimage=[]
                        landmark1=[]
                        img_rotate_90_clockwise=[]
                        img_rotate_90_clockwise1=[]
                        landmark=[]
                        landmark2=[]
                        img_rotate_90_counterclockwise=[]
                        cim=[]
                        cim1=[]
                        cim2=[]
                        cim3=[]
                        for num_s in range(len(sempic)-1):
                            if num not in sempic[num_s] and num in sempic[num_s + 1]:
                                newimage.append(num_s + 1)
                                cim.append((num_s + 1)*len(im)/len(sempic))
                            elif num in sempic[num_s] and num not in sempic[num_s + 1]:
                                newimage.append(num_s)
                                cim.append(num_s*len(im)/len(sempic))
                        #始点と終点の間をすべてリストの中に別々で格納
                        for n_s in range(len(newimage)//2):
                            landmark1.append(sempic[newimage[n_s*2]:newimage[n_s*2+1]])
                            cim1.append(im[int(cim[n_s*2]):int(cim[n_s*2+1])])
                        #リストの中身をすべて90度回転したものを新しいリストの中へ
                        for l_l,l_c in zip(landmark1,cim1):
                            img_rotate_90_clockwise.append(cv2.rotate(np.array(l_l), cv2.ROTATE_90_CLOCKWISE))
                            img_rotate_90_clockwise1.append(cv2.rotate(np.array(l_c), cv2.ROTATE_90_CLOCKWISE))
                        #ランドマークの始点と終点を検出
                        for y1,y2 in zip(img_rotate_90_clockwise,img_rotate_90_clockwise1):
                            for num_s2 in range(len(y1)-1):
                                if num not in y1[num_s2] and num in y1[num_s2 + 1]:
                                    landmark.append(num_s2 + 1)
                                    cim2.append((num_s2 + 1)*len(im[0])/len(y1))
                                elif num in y1[num_s2] and num not in y1[num_s2 + 1]:
                                    landmark.append(num_s2)
                                    cim2.append(num_s2*len(im[0])/len(y1))
                            for n_s2 in range(len(landmark)//2):
                                landmark2.append(y1[landmark[n_s2*2]:landmark[n_s2*2+1]])
                                cim3.append(y2[int(cim2[n_s2*2]):int(cim2[n_s2*2+1])])
                            for l_l2 in cim3:
                                img_rotate_90_counterclockwise.append(cv2.rotate(np.array(l_l2), cv2.ROTATE_90_COUNTERCLOCKWISE))
                            for n_p2,l_l3 in enumerate(img_rotate_90_counterclockwise):
                                cv2.imwrite(cat +"_third" +str(n_p2) +".png",l_l3)

                                myOCR(l_l3)
            else:pass
                            
            #画像を作成する
            class_names_list, label_values = helpers.get_label_info(os.path.join("Engineering", "class_color.csv"))

            out_vis_image = helpers.colour_code_segmentation(sempic, label_values)
            #print(out_vis_image.shape)

            out_vis_image = out_vis_image.reshape((jy, jx, 3))
            out_vis_image = cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR)
            out_vis_image = cv2.resize(out_vis_image, size)
            im = cv2.resize(im, size)
            blended = cv2.addWeighted(src1=im,alpha=0.5,src2=out_vis_image,beta=0.5,gamma=0)
            cv2.imshow('image', blended)
            cv2.imwrite("test.png",out_vis_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()