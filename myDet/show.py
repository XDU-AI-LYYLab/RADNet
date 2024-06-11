import os
import cv2

#--------------------------------- label show -----------------------------------
# label = 'E:\dataset\DIOR\MiniDIOR\\12014.txt'
# file = open(label, 'r')
# lines = file.readlines()
# img = cv2.imread('E:\dataset\DIOR\MiniDIOR\\12014.png')
# for line in lines:
#     l = line.split(' ')
#
#     # xmin,ymin,xmax, ymax = int(float(l[0])), int(float(l[1])), int(float(l[2])), int(float(l[3]))
#     # print(l[4])
#     # if l[4] == "ship\n":
#     #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#     #----------------------------------------------------------------------------
#     xmin = min(int(float(l[0])), int(float(l[2])), int(float(l[4])), int(float(l[6])))
#     xmax = max(int(float(l[0])), int(float(l[2])), int(float(l[4])), int(float(l[6])))
#     ymin = min(int(float(l[1])), int(float(l[3])), int(float(l[5])), int(float(l[7])))
#     ymax = max(int(float(l[1])), int(float(l[3])), int(float(l[5])), int(float(l[7])))
#     # print(l)
#     # print(ymax, ymin, xmax, xmin)
#     if (ymax- ymin) <= 50:
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
#
# cv2.imwrite('E:\dataset\DIOR\MiniDIOR\\12014_label.png', img)
# cv2.imshow('', img)
# cv2.waitKeyEx()

# -------------------------------------det show------------------------------------------------
txt_dir = '/home/lyy/part_images/ohd/third/txt_Merge/' #---------------------modify
# fpn_deconv_txt_dir = './fpn_deconv_txt'

# fpn = os.listdir(fpn_txt_dir)
# # rfpn = os.listdir(fpn_deconv_txt_dir)

# classes = ['large-vehicle', 'ship', 'small-vehicle', 'storage-tank']
classes = ['plane', 'ship']

results = {}

for cls in classes:
    file = open(os.path.join(txt_dir, cls+'.txt'), 'r')
    lines = file.readlines()
    for line in lines:
        l = line.split(' ')
        img_file = l[0]
        xmin,ymin,xmax, ymax = int(float(l[2])), int(float(l[3])), int(float(l[4])), int(float(l[5]))
        score = float(l[1])
        if results.__contains__(img_file):
            obj = {"class":cls, "bbox":[xmin, ymin, xmax, ymax], 'score':score}
            results[img_file].append(obj)
        else:
            results[img_file] = []
            obj = {"class": cls, "bbox": [xmin, ymin, xmax, ymax], 'score':score}
            results[img_file].append(obj)
for k, v in results.items():
    img = cv2.imread('/home/lyy/dataset/OHD-SJTU-small/test/images/' + k +'.png')
    for obj in v:
        obj_xmin, obj_ymin, obj_xmax, obj_ymax = obj["bbox"]
        obj_score = str(round(obj["score"], 2))
        obj_label = obj["class"]
        cv2.rectangle(img, (obj_xmin, obj_ymin), (obj_xmax, obj_ymax), (0, 0, 255), 2)
        cv2.putText(img, obj_label, (obj_xmin, obj_ymin), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255),
                    thickness=1)
    # img = cv2.resize(img, (1000, 600))
    cv2.imwrite('/home/lyy/part_images/ohd/third/det_img/' + k +'_det.png', img) #--------------------modify
    # cv2.imshow(k, img)
    # cv2.waitKeyEx()