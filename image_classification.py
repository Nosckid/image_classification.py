# Import necessary libraries
from imageai.Classification import ImageClassification
import os

# Get the current working directory or execution path
execution_path = os.getcwd()

# Initialize an ImageClassifcation instance
prediction = ImageClassification()

# Set the model type to MobileNetV2
prediction.setModelTypeAsMobileNetV2()

# Set the path to the MobileNetV2 model path
prediction.setModelPath(os.path.join(execution_path, "mobilenet_v2-b0353104.pth"))

# Load the MobileNetV2 model
prediction.loadModel()

# Classify the given image and get the predictions and probabilities for the top 5 classes
predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5)

# Iterate over the predictions and probabilities and print them
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(f"{eachPrediction}: {eachProbability}")
