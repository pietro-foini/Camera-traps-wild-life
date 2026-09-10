import os

os.environ["CLASSIFIER_MODEL_PATH"] = "/home/pippo/classifier.tflite"
os.environ["CLASSIFIER_CLASSES_PATH"] = "/home/pippo/labels.json"
os.environ["DETECTOR_MODEL_PATH"] = "/home/pippo/detector.tflite"
