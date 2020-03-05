from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import sys
import os
import copy
import pickle
import argparse
import facenet
import align.detect_face



def train(train_dir,facenet_model='models/pretrained/graph.pb', model_save_path=None, classifier = 'KNN', 
    n_neighbors=None, knn_algo='ball_tree', verbose=False, allowed_extensions=['.jpg']):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """

    # Loop through each person in the training set
    # load facenet graph



    images = []
    y = []
    for class_dir in os.listdir(train_dir):
        file_list = []
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for file in os.listdir(os.path.join(train_dir, class_dir)):
            valid_file=False
            for extension in allowed_extensions:
                if file.endswith(extension):
                    valid_file=True
                    break
            if valid_file==True:
                file_list.append(os.path.join(train_dir, class_dir)+'/'+file)
        class_images,_ = load_and_align_data(file_list)
        for i in range(0,class_images.shape[0]):
            y.append(class_dir)
        images.append(class_images)
    images = np.concatenate(images) 

    #load facenet and get embeddings

    emb = get_embeddings(facenet_model,images)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(emb))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    
    if classifier == 'KNN':
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    elif classifier == 'SVC':
        clf = SVC()

    clf.fit(emb, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(clf, f)

    return clf



def load_and_align_data(image_paths, image_size=160, margin=44, gpu_memory_fraction=1):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    imgs_with_faces = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        imgs_with_faces.append(image)  
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size),  interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return(images,imgs_with_faces)

def load_facenet(tf_model_path):

    GRAPH_PB_PATH = tf_model_path
    fd_graph = tf.Graph()
    with fd_graph.as_default():
        with tf.Session() as sess:
            with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:        
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,name='')
    return(fd_graph)

def get_embeddings(graph,images):

    with fd_graph.as_default():
        with tf.Session() as sess:

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)


def load_and_get_embeddings(tf_model_path,images):

    GRAPH_PB_PATH = tf_model_path
    fd_graph = tf.Graph()
    with fd_graph.as_default():
        with tf.Session() as sess:
            with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:        
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,name='')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            emb = sess.run(embeddings, feed_dict=feed_dict)
    return(emb)




def predict(X,clf=None, model_path=None):

    if clf is None and model_path is None:
        raise Exception("Must supply a classifier either thourgh clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)

    y = clf.predict(X)
    return(y)


def get_faces(image_paths, image_size=70, margin=44, gpu_memory_fraction=1,detection_confidence=0.9):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  
    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    imgs_with_faces = []
    output = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        output.append([image,bounding_boxes])

    return output




def get_face_labels(data,max_clusters=20, opt_cluster_threshold=8):


    inertia = []
    n_clusters = []
    labels = []
    for n in range(1,max_clusters+1):
        kmeans = KMeans(n_clusters=n)
        kmeans = kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        n_clusters.append(n)
        labels.append(kmeans.labels_)

    for i in range(1,max_clusters-1):
    
        slope_prev = (inertia[i] - inertia[i-1])/ (n_clusters[i] - n_clusters[i-1])
        slope_next = (inertia[i+1] - inertia[i])/ (n_clusters[i+1] - n_clusters[i])
    
        if (slope_next-slope_prev) < opt_cluster_threshold:
            break

    return(labels[i-1])


def get_distinct_faces(embeddings,max_clusters=20, opt_cluster_threshold=8):

    n_clusters = get_optimal_clusters(embeddings,max_clusters, opt_cluster_threshold)

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans = kmeans.fit(embeddings)
    return(kmeans.cluster_centers_)


def load_mtcnn(gpu_memory_fraction=1):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return(pnet,rnet,onet)


def detect_faces_mtcnn(pnet,rnet,onet,image_paths, image_size=160, margin=44):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    

    tmp_image_paths=copy.copy(image_paths)
    img_list = []
    imgs_with_faces = []
    for image in tmp_image_paths:
        img = misc.imread(os.path.expanduser(image), mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
          image_paths.remove(image)
          print("can't detect face, remove ", image)
          continue
        imgs_with_faces.append(image)  
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size),  interp='bilinear')
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return(images,imgs_with_faces)


class Facenet_encoder():
    def __init__(self, tf_model_path):

        GRAPH_PB_PATH = tf_model_path
        fd_graph = tf.Graph()
        with fd_graph.as_default():
            with tf.Session() as sess:
                with tf.io.gfile.GFile(GRAPH_PB_PATH,'rb') as f:        
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def,name='')
        self.graph = fd_graph

    def get_embeddings(self,images):

        with self.graph.as_default():
            with tf.Session() as sess:

                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                emb = sess.run(embeddings, feed_dict=feed_dict)
        return(emb)


