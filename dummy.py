import cv2
import sys
import os
import numpy as np
import argparse
import util
import cluster_util as cutil


def get_image():
	pass

def to_save_or_not():
	pass



model_path = 'models/pretrained/graph.pb'
cluster_path = 'some_path'     
cluster_name =  'some_name'
folder_to_store_images = 'some_path_again'
threshold = 0.72

#Load our model
encoder = util.Facenet_encoder(model_path)
#Load a cluster or database of face vectors
cluster = cutil.Classification_cluster(cluster_path,cluster_name,folder_to_store_images)

while(1):

##### wait for an image ####

#get image
	image = get_image()

#encode image
	embedding = encoder.get_embeddings(image)

#add to cluster
	dist, grp = cluster.match_face(embedding)
#classify based on threshold	
    if dist > threshold:
        cluster.add_vec(embedding)
    else:
        cluster.add_vec(embedding,group_no=grp)

#check periodically if it is time to save the cluster
    is_it_time_to_save_the_cluster_? = to_save_or_not():
    pass

    if is_it_time_to_save_the_cluster_:
    	cluster.save()