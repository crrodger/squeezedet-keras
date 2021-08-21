# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:36:53 2018

@author: crrodger
"""

from main.model.squeezeDet import  SqueezeDet
from main.model.dataGenerator import generator_from_data_path, visualization_generator_from_data_path
import keras.backend as K
from keras import optimizers
import tensorflow as tf
from main.model.evaluation import evaluate
from main.model.visualization import  visualize
import cv2
import os
import time
import numpy as np
import argparse
from keras.utils import multi_gpu_model
from main.config.create_config import load_dict
import main.utils.utils as utils

#default values for some variables
#TODO: uses them as proper parameters instead of global variables
predict_images = '../../image_inputs'
checkpoint_dir = '../../scripts/log/checkpoints'
output_images = '../../image_scored'
TIMEOUT = 20
EPOCHS = 100
CUDA_VISIBLE_DEVICES = "0"
steps = None
GPUS = 1
STARTWITH = None
CONFIG = "../config/squeeze.config"
TESTING = False
BEST_MODEL = "model.61-3.40.hdf5" #Name of model file hdf5 that contains best perfomring model on validation


def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box


def filter_prediction(boxes, probs, cls_idx, config):
    """Filter bounding box predictions with probability threshold and
    non-maximum supression.
    
    Args:
      boxes: array of [cx, cy, w, h].
      probs: array of probabilities
      cls_idx: array of class indices
    Returns:
      final_boxes: array of filtered bounding boxes.
      final_probs: array of filtered probabilities
      final_cls_idx: array of filtered class indices
    """

    #check for top n detection flags
    if config.TOP_N_DETECTION < len(probs) and config.TOP_N_DETECTION > 0:
      order = probs.argsort()[:-config.TOP_N_DETECTION-1:-1]
      probs = probs[order]
      boxes = boxes[order]
      cls_idx = cls_idx[order]
      
    else:

      filtered_idx = np.nonzero(probs>config.PROB_THRESH)[0]
      probs = probs[filtered_idx]
      boxes = boxes[filtered_idx]
      cls_idx = cls_idx[filtered_idx]
    
    final_boxes = []
    final_probs = []
    final_cls_idx = []

    #go trough classes
    for c in range(config.CLASSES):
      idx_per_class = [i for i in range(len(probs)) if cls_idx[i] == c]

      #do non maximum suppresion
      keep = utils.nms(boxes[idx_per_class], probs[idx_per_class], config.NMS_THRESH)
      for i in range(len(keep)):
        if keep[i]:
          final_boxes.append(boxes[idx_per_class[i]])
          final_probs.append(probs[idx_per_class[i]])
          final_cls_idx.append(c)

    return final_boxes, final_probs, final_cls_idx

def filter_batch( y_pred,config):
    """filters boxes from predictions tensor
    
    Arguments:
        y_pred {[type]} -- tensor of predictions
        config {[type]} -- squeezedet config
    
    Returns:
        lists -- list of all boxes, list of the classes, list of the scores
    """




    #slice predictions vector
    pred_class_probs, pred_conf, pred_box_delta = utils.slice_predictions_np(y_pred, config)
    det_boxes = utils.boxes_from_deltas_np(pred_box_delta, config)

    #compute class probabilities
    probs = pred_class_probs * np.reshape(pred_conf, [config.BATCH_SIZE, config.ANCHORS, 1])
    det_probs = np.max(probs, 2)
    det_class = np.argmax(probs, 2)



    #count number of detections
    num_detections = 0


    all_filtered_boxes = []
    all_filtered_scores = []
    all_filtered_classes = [ ]

    #iterate batch
    for j in range(config.BATCH_SIZE):

        #filter predictions with non maximum suppression
        filtered_bbox, filtered_score, filtered_class = filter_prediction(det_boxes[j], det_probs[j],
                                                                          det_class[j], config)


        #you can use this to use as a final filter for the confidence score
        keep_idx = [idx for idx in range(len(filtered_score)) if filtered_score[idx] > float(config.FINAL_THRESHOLD)]

        final_boxes = [filtered_bbox[idx] for idx in keep_idx]

        final_probs = [filtered_score[idx] for idx in keep_idx]

        final_class = [filtered_class[idx] for idx in keep_idx]


        all_filtered_boxes.append(final_boxes)
        all_filtered_classes.append(final_class)
        all_filtered_scores.append(final_probs)


        num_detections += len(filtered_bbox)


    return all_filtered_boxes, all_filtered_classes, all_filtered_scores

def score():
    """
    Takes images from specified directory and tries to find bounding boxes for each image.
    This is the 'unseen' data, i.e. actual scoring run
    """

    #hide the other gpus so tensorflow only uses this one
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    #create config object
    cfg = load_dict(CONFIG)
    cfg.BATCH_SIZE = 1

    #instantiate model
    squeeze = SqueezeDet(cfg)

    model = squeeze.model

    squeeze.model.load_weights(checkpoint_dir + "/"+ BEST_MODEL)
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    #Process each image in turn generating prediction bounding boxes
    for img_name in sorted(os.listdir(predict_images)):
        print("Working on " + img_name)
        img_base, img_ext = img_name.split(".")
        img = cv2.imread(predict_images+"/"+img_name).astype(np.float32, copy=False)
        img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        img_mtrx = np.zeros((1, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, cfg.N_CHANNELS))
        img_mtrx[0] = np.asarray(img)

        results = model.predict(img_mtrx)
        boxes , classes, scores = filter_batch(results, cfg)
        
        for j, det_box in enumerate(boxes[0]):

            #transform into xmin, ymin, xmax, ymax
            det_box = bbox_transform_single_box(det_box)

            #add rectangle and text
            cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (0,0,255), 1)
#            cv2.putText(img, config.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(all_filtered_scores[i][j]) , (det_box[0], det_box[1]), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(img, cfg.CLASS_NAMES[classes[0][j]] + " " + str(scores[0][j]) , (det_box[0], det_box[1]), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
            img_name_out = output_images + "/" + img_base + "_scored_" + BEST_MODEL.split(".")[1] + "." + img_ext
            cv2.imwrite(img_name_out, img)
        
#        res_box = [bbox_transform_single_box(a) for a in boxes[0]]
#        print(res_box)


if __name__ == "__main__":

    #argument parsing
    parser = argparse.ArgumentParser(description='Evaluate squeezeDet keras checkpoints after each epoch on validation set.')
    parser.add_argument("--logdir", help="dir with checkpoints and loggings. DEFAULT: ./log")
    parser.add_argument("--val_img", help="file of full path names for the validation images. DEFAULT: img_val.txt")
    parser.add_argument("--val_gt", help="file of full path names for the corresponding validation gts. DEFAULT: gt_val.txt")
    parser.add_argument("--test_img", help="file of full path names for the test images. DEFAULT: img_test.txt")
    parser.add_argument("--test_gt", help="file of full path names for the corresponding test gts. DEFAULT: gt_test.txt")
    parser.add_argument("--steps",  type=int, help="steps to evaluate. DEFAULT: length of imgs/ batch_size")
    parser.add_argument("--gpu",  help="gpu to use. DEFAULT: 1")
    parser.add_argument("--gpus",  type=int, help="gpus to use for multigpu usage. DEFAULT: 1")
    parser.add_argument("--epochs", type=int, help="number of epochs to evaluate before terminating. DEFAULT: 100")
    parser.add_argument("--timeout", type=int, help="number of minutes before the evaluation script stops after no new checkpoint has been detected. DEFAULT: 20")
    parser.add_argument("--init" , help="start evaluating at a later checkpoint")
    parser.add_argument("--config",   help="Dictionary of all the hyperparameters. DEFAULT: squeeze.config")
    parser.add_argument("--testing",   help="Run eval on test set. DEFAULT: False")

    args = parser.parse_args()

    #set global variables according to optional arguments
    if args.logdir is not None:
        log_dir_name = args.logdir
        checkpoint_dir = log_dir_name + '/checkpoints'
        tensorboard_dir = log_dir_name + '/tensorboard_val'

    if args.val_img is not None:
        img_file = args.val_img
    if args.val_gt is not None:
        gt_file = args.val_gt

    if args.test_img is not None:
        img_file_test = args.test_img
    if args.test_gt is not None:
        gt_file_test = args.test_gt

    if args.gpu is not None:
        CUDA_VISIBLE_DEVICES = args.gpu
    if args.epochs is not None:
        EPOCHS = args.epochs
    if args.timeout is not None:
        TIMEOUT = args.timeout

    if args.gpus is not None:
        GPUS = args.gpus


        #if there were no GPUS explicitly given, take the last ones
        #the assumption is, that we use as many gpus for evaluation as for training
        #so we have to hide the other gpus to not try to allocate memory there
        if args.gpu is None:
            CUDA_VISIBLE_DEVICES = ""
            for i in range(GPUS, 2*GPUS):
                CUDA_VISIBLE_DEVICES += str(i) + ","
            print(CUDA_VISIBLE_DEVICES)

    if args.init is not None:
        STARTWITH = args.init

    if args.steps is not None:
        steps = args.steps

    if args.config is not None:
        CONFIG = args.config

    if args.testing is not None:
        TESTING = args.testing

    score()