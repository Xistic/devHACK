from imageai.Detection import ObjectDetection
import os
import tensorflow as tf

exec_path=os.getcwd()

detec=ObjectDetection()
detec.setModelTypeAsRetinaNet()

detec.setModelPath(os.path.join(exec_path,"resnet50_coco_best_v2.0.1.h5"))
detec.loadModel()

list=detec.detectObjectsFromImage(
    input_image=os.path.join(exec_path, "imag.jpg"),
    output_image_path=os.path.join(exec_path, "newimag.jpg")
    
    )
