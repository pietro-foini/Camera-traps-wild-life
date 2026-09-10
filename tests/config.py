import os

os.environ["DB_USER"] = "pippo"
os.environ["DB_PASSWORD"] = "baudo"
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "1234"
os.environ["DB_NAME"] = "sanremo"
os.environ["CLASSIFIER_MODEL_PATH"] = "/home/pippo/classifier.tflite"
os.environ["CLASSIFIER_CLASSES_PATH"] = "/home/pippo/labels.json"
os.environ["DETECTOR_MODEL_PATH"] = "/home/pippo/detector.tflite"
