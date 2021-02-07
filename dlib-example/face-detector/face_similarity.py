#!/usr/bin/env python3

import os
import sys

import dlib
import numpy

predictor_path = 'shape_predictor_5_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(-1)

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    image_file1, image_file2 = sys.argv[1], sys.argv[2]

    if not os.path.isfile(image_file1) or not os.path.isfile(image_file2):
        sys.exit(-1)

    img1 = dlib.load_rgb_image(image_file1)
    img2 = dlib.load_rgb_image(image_file2)

    dets1 = detector(img1, 1)
    dets2 = detector(img2, 1)

    if not dets1 or not dets2:
        print("no box.")
        sys.exit(0)

    print(f"boxs1: {dets1}, boxs2: {dets2}")

    det1, det2 = dets1[0], dets2[0]

    shape1, shape2 = sp(img1, det1), sp(img2, det2)

    face_descript1 = facerec.compute_face_descriptor(img1, shape1, 10, 0.25)
    face_descript2 = facerec.compute_face_descriptor(img2, shape2, 10, 0.25)

    vector1, vector2 = numpy.array(face_descript1).reshape(1, 128), numpy.array(face_descript2).reshape(1, 128)

    euc_distance = numpy.linalg.norm(vector1 - vector2)
    cos_similarity = vector1.dot(vector2.reshape(128, 1)) / (numpy.linalg.norm(vector1) * numpy.linalg.norm(vector2))

    print(f"欧氏距离: {euc_distance}, 余弦相似度: {cos_similarity}")

