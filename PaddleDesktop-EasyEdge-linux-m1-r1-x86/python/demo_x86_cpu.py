# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了一个完整的demo
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import multiprocessing
import sys

import cv2
import numpy as np
import BaiduAI.EasyEdge as edge
import logging

edge.Log.set_level(logging.INFO)

try:
    _model_dir = sys.argv[1]
    _test_file = sys.argv[2]
except Exception as e:
    print("Usage: python3 demo.py {model_dir} {image_name | sound_name}")
    exit(-1)
_test_round = 1


def test():
    """单进程"""
    line_color = (125, 175, 75)

    pred = edge.Program()
    pred.set_auth_license_key("这里填写序列号")

    pred.init(model_dir=_model_dir, device=edge.Device.CPU, engine=edge.Engine.PADDLE_FLUID, thread_num=1)


    if pred.model_type == edge.c.ModelType.SoundClassification:
        # 声音分类模型示例
        with open(_test_file, 'rb') as fin:
            sound = fin.read()
            res = pred.infer_sound(sound, threshold=None)
            print(res)

    else:
        # 视觉类模型示例
        img = cv2.imread(_test_file)
        for i in range(_test_round):
            res = pred.infer_image(img, threshold=None)
            h, w, _ = img.shape
            imgc = img.astype(np.float32)
            for r in res:
                if pred.model_type == edge.c.ModelType.ObjectDetection or pred.model_type == edge.c.ModelType.FaceDetection:
                    x1 = int(r['x1'] * w)
                    y1 = int(r['y1'] * h)
                    x2 = int(r['x2'] * w)
                    y2 = int(r['y2'] * h)
                    cv2.rectangle(imgc, (x1, y1), (x2, y2), line_color, 2)
                    cv2.putText(
                        img=imgc, text=r['label'], org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=line_color, thickness=1
                    )
                    if pred.model_type == edge.c.ModelType.FaceDetection and 'face_points' in r:
                        colors = [(0, 0, 225), (0, 255, 225), (255, 0, 225), (0, 255, 0), (255, 0, 0)]
                        for i, point in enumerate(r['face_points']):
                            cv2.circle(imgc, point, 1, colors[i % 5], 4)
                elif pred.model_type == edge.c.ModelType.ImageSegmentation:
                    # Draw bbox
                    x1 = int(r["location"]["left"])
                    y1 = int(r["location"]["top"])
                    w = int(r["location"]["width"])
                    h = int(r["location"]["height"])
                    x2 = x1 + w
                    y2 = y1 + h

                    cv2.rectangle(imgc, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(imgc, "{} score: {}".format(r["name"], round(float(r["score"]), 4)),
                                (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)

                    # Draw mask
                    random_color = np.array([np.random.random() * 255.0,
                                             np.random.random() * 255.0,
                                             np.random.random() * 255.0])

                    idx = np.nonzero(r['mask'])
                    # alpha
                    imgc[idx[0], idx[1], :] *= 1.0 - 0.8
                    imgc[idx[0], idx[1], :] += 0.8 * random_color
                elif pred.model_type == edge.c.ModelType.SemanticSegmentation:
                    np.random.seed(11)
                    COLORS = np.random.randint(0, 255, size=(255, 3), dtype="uint8")
                    mask = COLORS[r['mask']]
                    imgc = ((0.4 * imgc) + (0.6 * mask)).astype("uint8")
                    break
                elif pred.model_type == edge.c.ModelType.PoseEstimation:
                    p1 = (int(r[edge.c.Keys.K_X1] * w), int(r[edge.c.Keys.K_Y1] * h))
                    cv2.circle(imgc, p1, 4, (255, 0, 0), -1, 8)
                    for pair_id in r[edge.c.Keys.K_POINT_PAIRS]:
                        for r2 in res:
                            if r2[edge.c.Keys.K_POINT_ID] == pair_id:
                                p2 = (int(r2[edge.c.Keys.K_X1] * w), int(r2[edge.c.Keys.K_Y1] * h))
                                cv2.line(imgc, p1, p2, (0, 0, 255), 4)
                print(r)

            if pred.model_type != edge.c.ModelType.ImageClassification:
                # cv2.imshow("Edge", imgc)
                cv2.imwrite(_test_file + ".result.jpg", imgc)

    pred.close()


# def test_multi_ncs():
#     """同时使用多个设备"""
#     q = multiprocessing.Queue(maxsize=100)
#
#     img = cv2.imread(_test_file)
#
#     def _task(i):
#         print("Process %d started" % i)
#         pred = edge.Program()
#         pred.init(model_dir=_model_dir, device=edge.Device.CPU, engine=edge.Engine.PADDLE_FLUID)
#
#         while True:
#             try:
#                 img = q.get(block=True, timeout=1)
#                 print("Process", i, pred.infer_image(img))
#             except Exception as e:
#                 break
#
#         pred.close()
#
#     p1 = multiprocessing.Process(target=_task, args=[1])
#     p2 = multiprocessing.Process(target=_task, args=[2])
#
#     p1.start()
#     p2.start()
#
#     for i in range(_test_round):
#         q.put(img)
#
#     p1.join()
#     p2.join()


if __name__ == '__main__':
    test()
