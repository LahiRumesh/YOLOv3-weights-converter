import os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
import colorsys
import tensorflow.keras.backend as K
from yolo3.utils import letterbox_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_image='input.jpg'
model_path='serving/yolov3/1'
classes_path='data_classes.txt'
out_put_image='out.jpg'
input_image_size=(416,416)

def generate_image_data(image):
    model_image_size = input_image_size
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0) # Add batch dimension.
    tf_image_data=tf.convert_to_tensor(image_data)
    return tf_image_data


def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names



def run_model(img_path,model_path,classes_path,img_out):

    model = tf.saved_model.load(model_path)
    infer = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    image=Image.open(img_path)
    image_data = generate_image_data(image)
    image_shape=np.array([image.size[1], image.size[0]], dtype='float32').reshape(1, 2)
    tf_image_shape=tf.convert_to_tensor(image_shape)
    outputs  = infer(image_shape=tf_image_shape,input_1=image_data)

    all_boxes_k, all_scores_k, indices_k= outputs["yolonms_layer"], outputs["yolonms_layer_1"], outputs["yolonms_layer_2"]
    all_boxes=all_boxes_k.numpy()
    all_scores=all_scores_k.numpy()
    indices=indices_k.numpy()

    out_boxes, out_scores, out_classes = [], [], []
    for idx_ in indices[0]:
        out_classes.append(idx_[1])
        out_scores.append(all_scores[tuple(idx_)])
        idx_1 = (idx_[0], idx_[2])
        out_boxes.append(all_boxes[idx_1])

    class_names=get_classes(classes_path)
    hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    np.random.seed(10101)  
    np.random.shuffle(colors) 
    np.random.seed(None)  
 
    font = ImageFont.truetype('yolov3/font/FiraMono-Medium.otf')
    thickness = int((image_shape[0][0] + image_shape[0][1]) // 300)
    out_prediction = []
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]
        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])
        out_prediction.append([left, top, right, bottom, predicted_class, score])
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    image.save(img_out)



if __name__ == "__main__":

    run_model(input_image,model_path,classes_path,out_put_image)
    


