import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from io import StringIO
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()
import pandas as pd



import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--i_dir", required=True,
    help="path of input directory")
ap.add_argument("-o", "--o_dir", required=True,
    help="path of output directory ")
ap.add_argument("-c", "--output_csv", required=True,
    help="name csv file")

args = vars(ap.parse_args())
print(args['i_dir'],args['o_dir'],args['output_csv'])
if os.path.exists(args['i_dir']):
    pass
else:
    os.makedirs(args['i_dir'])

if os.path.exists(args['o_dir']):
    pass
else:
    os.makedirs(args['o_dir'])






df = pd.DataFrame({'Image no':[],
                    'Class ': [],
                    'x':[],
                    'y':[],
                    'score':[]})


#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'#'faster_rcnn_resnet101_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

NUM_CLASSES = 10


global detection_graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, graph):
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(cv2.resize(image , (300,300)), 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      #print(output_dict['detection_classes'])
      
      return output_dict

from tensorflow import keras
import tensorflow as tf
import numpy as np
#import log

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)




classes_list = ['x1','v','t','x','v1','t1']
with detection_graph.as_default():
  with tf.Session() as sess:
    for jj in os.listdir('just_testing/'):
      image_np = cv2.imread('just_testing/{}'.format(jj))
      img = image_np.copy()
      im_width,im_height,channels = image_np.shape

      output_dict = run_inference_for_single_image(image_np, detection_graph)
      vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)
      boxes = output_dict['detection_boxes']
      classes = output_dict['detection_classes']

      max_boxes_to_draw = boxes.shape[0]
      scores = output_dict['detection_scores']
      min_score_thresh=.5
      

      for i,b in enumerate(boxes[0]):
          if scores[i] > min_score_thresh:

            mid_x = (boxes[i][1]+boxes[i][3])/2
            mid_y = (boxes[i][0]+boxes[i][2])/2

            
            cv2.putText(image_np, '.', (int(mid_x*im_height),int(mid_y*im_width)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.rectangle(image_np, (int((mid_x*im_height)-30), int((mid_y*im_width)-30)), (int((mid_x*im_height)+30), int((mid_y*im_width)+30)), (255,0,0), 2)
            x,y,w,h = int((mid_x*im_height)-30), int((mid_y*im_width)-30), int((mid_x*im_height)+30), int((mid_y*im_width)+30)
            new_image = img[y:h,x:w]
            


            #cv2.imshow('n',new_image)
            #cv2.circle(image_np,(int(mid_x*im_height),int(mid_y*im_width)),3,255,3)
            print(x,y,w,h)

            #cv2.imwrite('out/{}-{}.jpg'.format(jj[:-4],i),new_image)

            cur_class = classes_list[classes[i]-1]
            df.loc[len(df)] = [jj,cur_class,mid_x,mid_y,scores[i]]
            df.to_csv('{}/{}'.format(args['o_dir'],args['output_csv']),index=False)



      print('_______________________')

      cv2.imshow('object detection', image_np)



      cv2.waitKey(0)
    '''
      key = cv2.waitKey(33)
      if key==27:
        cv2.destroyWindow()
        break
    '''











#cv2.rectangle(image_np, (int(x), int(y)), (int(w), int(h)), (255,0,0), 2)
