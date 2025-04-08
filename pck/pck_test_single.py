from ultralytics import YOLO
import scipy.io
import numpy as np


def run_model(model_name, path):

    # Run tracking
    model = YOLO(model_name)
    results = model.track(source=path, show=False, save=False)

    # Define the remapping from YOLO keypoints (0-16) to indices
    keypoint_mapping = {5: 13, 6: 12, 7: 14, 8: 11, 9: 15, 10: 10, 11: 3, 12: 2, 13: 4, 14: 1, 15: 5, 16: 0}

    # Extract and format keypoints
    predictions = {}

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy  # Get keypoints in (x, y) format
            for yolo_index, (x, y) in enumerate(keypoints[0]):  # Assume single detection
                if yolo_index in keypoint_mapping:
                    custom_index = keypoint_mapping[yolo_index]
                    predictions[custom_index] = (float(x), float(y))  # Convert tensor values to floats

    return predictions


def mat_struct_to_dict(mat_struct):
    """ Recursively converts a MATLAB struct to a Python dictionary """
    if isinstance(mat_struct, np.ndarray):
        return [mat_struct_to_dict(item) for item in mat_struct]
    elif hasattr(mat_struct, '_fieldnames'):
        return {field: mat_struct_to_dict(getattr(mat_struct, field)) for field in mat_struct._fieldnames}
    else:
        return mat_struct


def extract_joints(mat_file, target_image):
    """
    Extracts the specified joints from the dataset for a given image.
    """
    joint_order = [13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]

    data = scipy.io.loadmat(mat_file, struct_as_record=False, squeeze_me=True)
    release = mat_struct_to_dict(data['RELEASE'])
    annolist = release['annolist']

    for img_data in annolist:
        if img_data['image']['name'] == target_image:
            joints = {}
            annorect = img_data.get('annorect', [])
            if isinstance(annorect, dict):
                annorect = [annorect]

            for person in annorect:
                annopoints = person.get('annopoints', {}).get('point', [])
                if isinstance(annopoints, dict):
                    annopoints = [annopoints]

                for point in annopoints:
                    joint_id = point['id']
                    if joint_id in joint_order:
                        joints[joint_id] = (point['x'], point['y'], point.get('is_visible', 1))

            # Reorder joints according to joint_order
            ordered_joints = {joint: joints[joint] for joint in joint_order if joint in joints}
            return ordered_joints

    return {}


def calculate_pck(predictions, ground_truth, threshold):
    """
    Calculates the PCK@0.5 score between predicted and ground truth keypoints.
    """
    pred_points = []
    gt_points = []
    visibilities = []

    for joint in ground_truth:
        x_gt, y_gt, visible = ground_truth[joint]
        if visible == 0:
            continue  # Skip joints that are not visible

        if joint in predictions:
            x_pred, y_pred = predictions[joint]
            pred_points.append((x_pred, y_pred))
            gt_points.append((x_gt, y_gt))
            visibilities.append(visible)

    if not pred_points:
        return None  # Return None if there are no visible joints

    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)

    # Compute normalization factor (bounding box diagonal)
    bbox_diagonal = np.linalg.norm(gt_points.max(axis=0) - gt_points.min(axis=0))

    # Compute distances
    distances = np.linalg.norm(pred_points - gt_points, axis=1) / bbox_diagonal

    # Compute PCK
    correct_keypoints = np.sum(distances < threshold)
    return correct_keypoints / len(gt_points)


# Run
mat_file_path = "mpii_human_pose_v1_u12_1.mat"
target_image = "000001163.jpg"
target_path = "data/images-small/000001163.jpg"
ground_truth = extract_joints(mat_file_path, target_image)
# model_name = "yolov8n-pose.pt"
model_name = "yolo11n-pose.pt"

predictions = run_model(model_name, target_path)

pck_threshold = 0.5
pck_score = calculate_pck(predictions, ground_truth, pck_threshold)
print(pck_threshold)
print(f"PCK@{pck_threshold} Score:", pck_score)