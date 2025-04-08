import os
import pickle
import pck_test_single
import time

from pck_test_single import pck_threshold
from pck_test_single import model_name

# Paths
# test_folder = "data/images-small"

# test_folder = "data/noisy_images2-5"
# test_folder = "data/noisy_images5"
test_folder = "data/noisy_images10"
# test_folder = "data/noisy_images"






# test_folder = "data/lens_distorted_images"
# test_folder = "data/noisy_images"
# test_folder = "data/perspective_distorted_images"
# test_folder = "data/rolling_shutter_output"


pickle_file = "joints_data_2.pkl"  # The file where extracted joints are stored

# Load pre-saved joint data from the pickle file
with open(pickle_file, "rb") as f:
    joints_dict = pickle.load(f)

total_images = 0
total_pck = 0
start_time = time.time()
current_amount = 0

for file in os.scandir(test_folder):
    current_amount += 1
    if current_amount % 10 == 0:
        print(current_amount)
    if file.is_file():
        file_path = file.path
        if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image_name = os.path.basename(file_path)
            print(image_name)

            # Get ground truth joints from the loaded dictionary
            ground_truth = joints_dict.get(image_name, None)

            if ground_truth is None:
                print(f"No ground truth found for {image_name}, skipping...")
                continue

            try:
                predictions = pck_test_single.run_model(model_name, file_path)
                pck = pck_test_single.calculate_pck(predictions, ground_truth, pck_threshold)
                print(f"PCK for {image_name}: {pck}")
            except AttributeError:
                print(f"Error processing {image_name}, skipping...")
                continue

            if pck is not None:
                total_pck += pck
                total_images += 1

end_time = time.time()

# Final results
output_folder = ""
output_path = os.path.join(output_folder, test_folder)


print("Total time:", end_time - start_time)
print("Total images:", total_images)
# print("Total pck:", total_pck)
print(f"Average PCK@:{pck_threshold} using model {model_name} =", total_pck / total_images if total_images > 0 else "N/A")

# with open(output_path, "w") as f:
#     f.write(f"Total time: {end_time - start_time}")
#     f.write(f"Total images: {total_images}")
#     f.write(f"Total mse: {total_mse}")
#     f.write(f"Average mse: {total_mse/total_images}")