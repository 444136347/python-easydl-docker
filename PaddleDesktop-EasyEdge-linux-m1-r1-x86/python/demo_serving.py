# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
本文件实现了一个http serving服务的demo
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

import sys
import BaiduAI.EasyEdge as edge
from BaiduAI.EasyEdge.serving import Serving

import logging

edge.Log.set_level(logging.INFO)

try:
    _model_dir = sys.argv[1]
except Exception as e:
    print("Usage: python3 demo_serving.py {model_dir} {serial_key} {host} {port}")
    exit(-1)


def test():
    """
    http serving
    :return:
    """
    arg_num = len(sys.argv)
    serial_key = ""
    host = "0.0.0.0"
    port = "24401"
    if arg_num >= 3:
        serial_key = sys.argv[2]  # 序列号
    if arg_num >= 4:
        host = sys.argv[3]
    if arg_num >= 5:
        port = sys.argv[4]

    server = Serving(model_dir=_model_dir, license=serial_key)
    # 请参考同级目录下demo.py里:
    # pred.init(model_dir=xx, device=xx, engine=xx, device_id=xx)
    # 对以下参数device\device_id和engine进行修改
    server.run(host=host, port=port, device=edge.Device.CPU, engine=edge.Engine.PADDLE_FLUID)


if __name__ == '__main__':
    test()


# ===========可打开浏览器更直观使用http服务，也可以参考以下代码使用==========
# def http_client_test():
#     import requests
#     url = 'http://0.0.0.0:24401/'
#     thresh = 0.4
#
#     try:
#         # 图像预测示例
#         import cv2
#         img = cv2.imread('图像路径')
#         ret, buffer = cv2.imencode('.jpg', img)
#         data = buffer.tobytes()
#         # 请求预测服务
#         response = requests.post(url, params={'threshold': thresh}, data=data).json()
#         return response # <class 'dict'>
#     except Exception as e:
#         print(e)
#         return {}
#
#     # 目标跟踪预测示例
#     try:
#         video_path = '视频路径'
#         with open(video_path, 'rb') as f:
#             data = f.read()
#         # 请求预测服务
#         # resetTracker: 1-表示需要重置, tracker 0-表示不用重置, 默认为1
#         response = requests.post(url, params={'threshold': thresh, 'resetTracker': 1}, data=data).json()
#         return response
#     except Exception as e:
#         print(e)
#         return {}
