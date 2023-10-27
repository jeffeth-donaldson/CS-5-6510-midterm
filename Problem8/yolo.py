import os,time
import numpy as np
from ultralytics import YOLO

# Load models
yolo8 = YOLO("yolov8n.pt")
yolo5 = YOLO("yolov5n.pt")
# Use the model
test_path = 'data/grocerystore'
validation_img_paths = [f'{test_path}/{f}' for f in os.listdir(test_path) if os.path.isfile(os.path.join(test_path, f))][:100]

v8times = []
v5times = []
for i in range(5):
    start = time.time()
    res5=yolo5.predict(validation_img_paths)
    v5times.append(time.time()-start)
    # 14.06 seconds, 7.1 fps, 61% CPU usage peak

    start = time.time()
    res8=yolo8.predict(validation_img_paths)
    v8times.append(time.time()-start)
    # 15.24 seconds, 6.6fps, 56% CPU usage peak
print('v8 avg runtime:',np.mean(v8times),'fps:',100/np.mean(v8times))
print('v5 avg runtime:',np.mean(v5times),'fps:',100/np.mean(v5times))

