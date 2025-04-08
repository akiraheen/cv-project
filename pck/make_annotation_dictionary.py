import os
import pickle
import scipy.io
import numpy as np
import time

start_time = time.time()


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
                try:
                    annopoints = person.get('annopoints', {}).get('point', [])
                except AttributeError:
                    continue

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


def process_images(mat_file, images_folder, output_file):
    """
    Loops through the images folder, extracts joints for each image, and saves to a pickle file.
    """
    image_joints_dict = {}
    image_count = 0

    for image_file in os.listdir(images_folder):
        image_count += 1
        if image_count % 10 == 0:
            print("Image count {}".format(image_count))

        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Ensure only images are processed
            joints = extract_joints(mat_file, image_file)
            if joints:  # Only save images that have joint data
                image_joints_dict[image_file] = joints

            else:
                print('Skipping {}'.format(image_file))


    # Save to pickle file
    with open(output_file, "wb") as f:
        pickle.dump(image_joints_dict, f)

    print(f"Saved joint data for {len(image_joints_dict)} images to {output_file}")

# Run
mat_file_path = "mpii_human_pose_v1_u12_1.mat"
images_folder_path = "data/images-small"
output_pickle_file = "joints_data_2.pkl"

process_images(mat_file_path, images_folder_path, output_pickle_file)

print("Total time: ", time.time() - start_time)