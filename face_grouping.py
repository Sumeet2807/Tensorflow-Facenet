import cv2
import sys
import os
import numpy as np
import argparse
import util
import shutil


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
ap.add_argument("-o", "--output", required=True, default=None,
    help="path to output image")
ap.add_argument("-m", "--model", default="models/pretrained/graph.pb",
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
    help="minimum probability to filter weak detections")
ap.add_argument("-mc", "--max_clusters", default=20,
    help="path to input model")
ap.add_argument("-oct", "--threshold", default=8,
    help="path to input model")
args = vars(ap.parse_args())


file_list = []
allowed_extensions = ['.jpg']

for file in os.listdir(args['image']):
    valid_file=False
    for extension in allowed_extensions:
        if file.endswith(extension):
            valid_file=True
            break
    if valid_file==True:
        file_list.append(args['image'] +'/'+file)
print("detecting faces")        
images,img_with_faces = util.load_and_align_data(file_list)
print("getting embeddings")
embeddings = util.get_embeddings(args['model'],images)
print('finding distinct groups')
faces = util.get_face_labels(embeddings,max_clusters=int(args['max_clusters']), opt_cluster_threshold=int(args['threshold']))
if args['output'] != None:
    i=0
    for img in img_with_faces:
        person_name = 'person' + str(faces[i])
        cls_folder = args['output'] + '/' + person_name
        if not os.path.isdir(cls_folder):
            os.makedirs(cls_folder) 
        new_path = cls_folder + '/' + str(i) + '.jpg'
        shutil.move(img, new_path)
        i += 1



