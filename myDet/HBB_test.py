#------------------------------------------------HBB DIOR dataset------------------------------------
import os
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result

import mmcv
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'
config_file = './ohd_configs/faster_rcnn_RFPN.py'
checkpoint_file = './tools/work_dirs/OHD/skip_faster_rcnn_r50_rfpn(resize_conv)_1x/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

score_thr = 0.3

class_name = model.CLASSES
print(class_name)
# test a list of images and write the results to image files
img_path = '/home/lyy/dataset/OHD-SJTU/test/images/' #------------------------ohd
# img_path = '/home/lyy/dataset/DIOR/Test/images/' #---------------------dior
# img_path = '/home/lyy/dataset/miniDOTA/val_sp_200_1024/images/' #----------------sdota
#---------------------------------det  imgae save-------------------------------------
save_path = '/home/lyy/part_images/ohd/first/det/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# ==================================== 创建 txt保存的 文件夹 ===============
# txt_dir = './tools/work_dirs/miniDOTA/skip_faster_rcnn_r101_fpn_(resize_conv)/results/'
# txt_dir = '/home/lyy/part_images/ohd/third/txt/'
# isExitdir = os.path.exists(txt_dir)
# if not isExitdir:
#     os.makedirs(txt_dir)
#====================================================================
imgs = [img_path + i for i in os.listdir(img_path)]
# print(imgs)
pbar = tqdm(total = len(os.listdir(img_path)))
for i, image in enumerate(imgs):
    # print(image)
    result = inference_detector(model, image)
    # show_result(imgs[i], result, model.CLASSES, out_file='result_{}.png'.format(i))
    img = mmcv.imread(image)
    detection = []
    img_name = os.path.basename(imgs[i]).replace('.png', '')

    out_file = os.path.join(save_path, '{}.png'.format(img_name))
    # out_file = None
    # print(result)
    if isinstance(result, tuple):  # 每张图的检测框
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None
        # print(bbox_result)
    for label in range(len(bbox_result)):  # 每张图中每一类对应的标签框
        # print(label)
        bboxes = bbox_result[label]
        if len(bboxes) == 0:
                continue
        inds = np.where(bboxes[:, -1] >=score_thr)[0]

        for i in inds:
            det = dict()
            det['score'] = bboxes[i, -1]
            det['class_name'] = class_name[label]
            det['boxes'] = [bboxes[i, 0], bboxes[i, 1], bboxes[i, 2], bboxes[i, 3]]
            detection.append(det)

#---------------------------------------------write -------------------------------------------------------
    # write_handle = {}
    # for sub_class in class_name:
    #     write_handle[sub_class] = open(os.path.join(txt_dir, '%s.txt' % sub_class), 'a+')
    # for i, bbox in enumerate(detection):
    #     command = '%s %.3f %.1f %.1f %.1f %.1f \n' % (img_name, detection[i]['score'],
    #                                                                      detection[i]['boxes'][0],
    #                                                                      detection[i]['boxes'][1],
    #                                                                      detection[i]['boxes'][2], detection[i]['boxes'][3])
    #     write_handle[detection[i]['class_name']].write(command)
    # for sub_class in class_name:
    #     write_handle[sub_class].close()
#----------------------------------------------------------------------------------------------------------

    pbar.set_description('Test image %s'% img_name)
    pbar.update(1)
#----------------------------------------------show--------------------------------------------------------
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]

    labels = np.concatenate(labels)
    mmcv.imshow_det_bboxes(
        img.copy(),
        np.vstack(bbox_result),
        labels,
        class_names=class_name,
        score_thr=score_thr,
        show=False,
        wait_time=0,
        out_file=out_file)
