import torch
from ultralytics import YOLO
import torchvision.models as models

# results11 = model11.track(source="data/000041029.jpg", show=True, save=True)
#
model11 = YOLO("yolo11n-pose.pt")
results11 = model11.track(source="data/images/000022704.jpg", show=True, save=True)

print("Printing yolo11 model.model.named_modules()")

for i, (name, module) in enumerate(model11.model.named_modules()):
    print(f"Name = {name}")

model8 = YOLO("yolov8n-pose.pt")
results8 = model8.track(source="data/images/000022704.jpg", show=True, save=True)

print("Printing yolov8 model.model.named_modules()")

for i, (name, module) in enumerate(model8.model.named_modules()):
    print(f"Name = {name}")


#
# # Inspect the results structure
# print(f"Type of results: {type(results11)}")
# print(f"Length of results: {len(results11)}")
#
# # Inspect the first result
# first_result = results11[0]
# print(f"Type of first result: {type(first_result)}")
# print(f"Available attributes: {dir(first_result)}")
#
# # Check keypoints specifically
# if hasattr(first_result, 'keypoints'):
#     print(f"Type of keypoints: {type(first_result.keypoints)}")
#     print(f"Keypoints shape: {first_result.keypoints.shape if hasattr(first_result.keypoints, 'shape') else 'N/A'}")
#     print(f"Keypoints data example: {first_result.keypoints.data if hasattr(first_result.keypoints, 'data') else first_result.keypoints}")
#
# # For more detailed inspection
# print("\nFull keypoints structure:")
# print(first_result.keypoints)