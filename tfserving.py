import sys
import tensorflow as tf
import time
import requests
import json

size = 320

start_time = time.time()

image = tf.image.decode_image(open(sys.argv[1], 'rb').read(), channels=3)
image = tf.expand_dims(image, axis=0)
image = tf.image.resize(image, (size, size))
image = image / 255

data = {
    "signature_name": "serving_default",
    "instances": image.numpy().tolist()
}
resp = requests.post("http://localhost:8501/v1/models/yolov3:predict", json=data)
resp = json.loads(resp.content.decode('utf-8'))['predictions'][0]

valid_predictions = resp['yolo_nms_3']
for i in range(valid_predictions):
    clazz = resp['yolo_nms_2'][i]
    print("detected", clazz)

print("took", time.time() - start_time, "seconds")
