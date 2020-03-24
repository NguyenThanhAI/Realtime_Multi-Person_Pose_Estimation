import os
import argparse
import time
import json
import numpy as np
import cv2
import caffe

from testing.python.processing import extract_parts, draw
from testing.python.config_reader import config_reader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')

    args = parser.parse_args()

    image_path = args.image

    output = os.path.join(os.path.dirname(image_path), os.path.basename(image_path).split(".")[0] + "_" + args.output)

    keypoint_to_label = {'nose': 0, 'neck': 1, 'right_shoulder': 2, 'right_elbow': 3, 'right_wrist': 4,
                         'left_shoulder': 5, 'left_elbow': 6, 'left_wrist': 7, 'right_hip': 8, 'right_knee': 9,
                         'right_ankle': 10, 'left_hip': 11, 'left_knee': 12, 'left_ankle': 13, 'right_eye': 14,
                         'left_eye': 15, 'right_ear': 16, 'left_ear': 17}
    label_to_keypoint = {v: k for k, v in keypoint_to_label.items()}

    print('start processing...')

    param, model = config_reader()

    if param['use_gpu']:
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber'])  # set to your device!
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)

    tic = time.time()

    input_image = cv2.imread(image_path)  # B,G,R order
    height, width = input_image.shape[:2]

    body_parts, all_peaks, subset, candidate = extract_parts(input_image, param, net, model)
    # print("body_parts:", body_parts, "all_peaks:", len(all_peaks), "subset:", subset, "candidate:", candidate)
    canvas = draw(input_image, all_peaks, subset, candidate)

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))


    # cv2.imshow("", draw_image)
    # cv2.waitKey(0)

    cv2.imwrite(output, canvas)

