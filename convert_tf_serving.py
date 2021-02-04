import os
from timeit import default_timer as timer
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body,yolo_boxes_and_scores
from yolo3.utils import letterbox_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


model_path='weights.h5'
anchors_path='main_anchors.txt'
classes_path='data_classes_2.txt'
score_threshold=0.3
iou_threshold=0.1               
output_model='serving/yolov3/1'

class YOLOEvaluationLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(YOLOEvaluationLayer, self).__init__()
        self.anchors = np.array(kwargs.get('anchors'))
        self.num_classes = kwargs.get('num_classes')

    def get_config(self):
        config = {
            "anchors": self.anchors,
            "num_classes": self.num_classes,
        }

        return config

    def call(self, inputs, **kwargs):
        """Evaluate YOLO model on given input and return filtered boxes."""
        yolo_outputs = inputs[0:-1]
        input_image_shape = K.squeeze(inputs[-1], axis=0)
        num_layers = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5],
                                                                                 [1, 2, 3]]  # default setting
        input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        boxes = []
        box_scores = []
        for l in range(num_layers):
            _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], self.anchors[anchor_mask[l]], self.num_classes,
                                                        input_shape, input_image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = K.concatenate(boxes, axis=0)
        box_scores = K.concatenate(box_scores, axis=0)
        return [boxes, box_scores]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, 4), (None, None)]


class YOLONMSLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(YOLONMSLayer, self).__init__()
        self.max_boxes = kwargs.get('max_boxes', 20)
        self.score_threshold = kwargs.get('score_threshold', score_threshold)
        self.iou_threshold = kwargs.get('iou_threshold', iou_threshold)
        self.num_classes = kwargs.get('num_classes')

    def get_config(self):
        config = {
            "max_boxes": self.max_boxes,
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "num_classes": self.num_classes,
        }

        return config

    def call(self, inputs, **kwargs):
        boxes = inputs[0]
        box_scores = inputs[1]
        box_scores_transpose = tf.transpose(box_scores, perm=[1, 0])
        boxes_number = tf.shape(boxes)[0]
        box_range = tf.range(boxes_number)

        mask = box_scores >= self.score_threshold
        max_boxes_tensor = K.constant(self.max_boxes, dtype='int32')
        classes_ = []
        batch_indexs_ = []
        nms_indexes_ = []
        class_box_range_ = []
        for c in range(self.num_classes):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            class_box_range = tf.boolean_mask(box_range, mask[:, c])
            nms_index = tf.image.non_max_suppression(
                class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=self.iou_threshold)
            class_box_scores = K.gather(class_box_scores, nms_index)
            class_box_range = K.gather(class_box_range, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c
            batch_index = K.zeros_like(class_box_scores, 'int32')
            batch_indexs_.append(batch_index)
            classes_.append(classes)
            nms_indexes_.append(nms_index)
            class_box_range_.append(class_box_range)

        classes_ = K.concatenate(classes_, axis=0)
        batch_indexs_ = K.concatenate(batch_indexs_, axis=0)
        class_box_range_ = K.concatenate(class_box_range_, axis=0)

        boxes_1 = tf.expand_dims(boxes, 0)
        classes_1 = tf.expand_dims(classes_, 1)
        batch_indexs_ = tf.expand_dims(batch_indexs_, 1)
        class_box_range_ = tf.expand_dims(class_box_range_, 1)
        box_scores_transpose_1 = tf.expand_dims(box_scores_transpose, 0)
        nms_final_ = K.concatenate([batch_indexs_, classes_1, class_box_range_], axis=1)
        nms_final_1 = tf.expand_dims(nms_final_, 0)
        return [boxes_1, box_scores_transpose_1, nms_final_1]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [(None, None, 4), (None, self.num_classes, None), (None, None, 3)]


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt'
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        start = timer()
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        end = timer()

        input_image_shape = tf.keras.Input(shape=(2,), name='image_shape')
        image_input = tf.keras.Input((None, None, 3), dtype='float32', name='input_1')
        y = list(self.yolo_model(image_input))
        y.append(input_image_shape)

        boxes, box_scores = \
            YOLOEvaluationLayer(anchors=self.anchors, num_classes=len(self.class_names))(inputs=y)

        out_boxes, out_scores, out_indices = \
            YOLONMSLayer(anchors=self.anchors, num_classes=len(self.class_names))(
                inputs=[boxes, box_scores])
        self.final_model = tf.keras.Model(inputs=[image_input, input_image_shape],
                                       outputs=[out_boxes, out_scores, out_indices])

        tf.saved_model.save(self.final_model,output_model)
        print('{} model, anchors, and classes loaded and model converted Sucessfully !!! {:.2f}sec.'.format(model_path, end-start))




if __name__ == "__main__":
    yolo = YOLO(**{"model_path": model_path,
                "anchors_path": anchors_path,
                "classes_path": classes_path
                }
               )