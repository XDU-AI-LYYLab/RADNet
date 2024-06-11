import cv2
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
imgpath = '/home/lyy/hq/dataSet/miniDOTA/part3/12060.png' #minidota
# imgpath = '/home/lyy/hq/DTestRes/dior_dota/part_images/12014.png'
# imgpath = '/home/lyy/hq/imgs/P0097__1__0___0.png' #环路
# imgpath = '/home/lyy/hq/imgs/P0112__1__824___3296.png' #游泳池




def draw_features(channel, width, height, x, savename, savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    image = cv2.imread(imgpath)
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    feature_map_combination = []
    for i in range(channel):
        # plt.subplot(height, width, i + 1)
        # plt.axis('off')
        img = x[0, i, :, :]
        feature_map_combination.append(img)
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        # img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        # img_test2 = cv2.resize(img, (0, 0), fx=256, fy=256, interpolation=cv2.INTER_NEAREST)
        # plt.imshow(img_test2)

        tmp_file = os.path.join(savepath, savename + '_' + str(i) + '.png')
        tmp_img = img.copy()
        tmp_img = cv2.resize(tmp_img, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)

        temp = cv2.addWeighted(tmp_img, 0.7, image, 0.3, 50)
        cv2.imwrite(tmp_file, temp)

        # if img.shape[0] < 256:
        #     tmp_file = os.path.join(savepath, savename + '_' + str(i)  + '.png')
        #     tmp_img = img.copy()
        #     tmp_img = cv2.resize(tmp_img, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
        #
        #     temp = cv2.addWeighted(tmp_img, 0.6, image, 0.4, 50)
        #     cv2.imwrite(tmp_file, temp)

    # feature_map_sum = sum(ele for ele in feature_map_combination)
    # pmin = np.min(feature_map_sum)
    # pmax = np.max(feature_map_sum)
    # feature_map_sum = ((feature_map_sum - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
    # feature_map_sum = feature_map_sum.astype(np.uint8)  # 转成unit8
    # feature_map_sum = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)  # 生成heat map
    # temp_img = cv2.resize(feature_map_sum, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    #
    # temp = cv2.addWeighted(temp_img, 0.5, image, 0.5, 50)
    #
    # cv2.imwrite(os.path.join(savepath, savename +'_sum.png'), temp)
    # print("time:{}".format(time.time() - tic))


import cv2
from pylab import *




def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(img_batch,savename, savepath):
    img = cv2.imread(imgpath)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # print(img_batch.shape)
    feature_map = np.squeeze(img_batch, axis=0)
    # print(feature_map.shape)
    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[0]
    row, col = get_row_col(num_pic)
    for i in range(0, num_pic):
        feature_map_split = feature_map[i, :, :]

        tmp_img = cv2.resize(feature_map_split, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        feature_map_combination.append(tmp_img)
        # plt.subplot(row, col, i + 1)
        plt.imshow(tmp_img)
        axis('off')
        # title('feature_map_{}'.format(i))

        tmp_file = os.path.join(savepath, savename + '_' + str(i) + '.png')

        plt.savefig(tmp_file)


    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    # feature_map_sum /= len(feature_map_combination)
    plt.imshow(feature_map_sum)
    file = os.path.join(savepath,savename+"_sum.png")
    plt.savefig(file)


