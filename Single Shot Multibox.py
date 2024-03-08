import tensorflow as tf
import cv2

def detect_objects_ssd(image_path, model_path):
    model = tf.saved_model.load(model_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (300, 300))
    image = image[..., ::-1].astype('float32') / 255.0
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

# Example usage:
image_path = "image.jpg"
model_path = "ssd_model"
objects = detect_objects_ssd(image_path, model_path)
print("Detected objects:", objects)
