# YOLOv3 weights (Keras H5 format) convert for Tensorflow Serving

This can use to convert Tensorflow(Keras H5 format) for Tensorflow Serving


#### The test environment 

    - Python 3.7
    - tensorflow 2.3.1
    - numpy 1.18.5
    - Pillow 8.0.1


####  Convert yolov3.h5 weights  

Make changes to **convert_tf_serving.py** , 

```bash
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #select the GPU 

model_path='weights.h5'  
anchors_path='main_anchors.txt'
classes_path='data_classes.txt'
score_threshold=0.5 
iou_threshold=0.25               
output_model='serving/yolov3/1'

```

####  Test Tensorflow Serving Model  

```bash
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #select the GPU 

input_image='1.jpg'
model_path='serving/yolov3/1'
classes_path='data_classes.txt'
out_put_image='out.jpg'
input_image_size=(416,416)

```


## Acknowledgements

 [keras-yolo3](https://github.com/qqwweee/keras-yolo3) for more information.
