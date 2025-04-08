import torchvision.transforms as T
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import entropy, skew, kurtosis


def extract_feature_maps(model, image_path, layer_name, save_path="NEW"):

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)  # Shape: (1, C, H, W)

    feature_maps = []

    # hook to capture feature maps
    def hook_fn(module, input, output):
        feature_maps.append(output.detach().cpu().squeeze(0))  # Remove batch dim

    target_layer = dict(model.model.named_modules()).get(layer_name, None)
    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in the model")

    hook = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(input_tensor)

    hook.remove()

    if not feature_maps:
        print("no feature maps found!!")
        return None

    feature_maps = feature_maps[0]

    # plots
    num_channels = feature_maps.shape[0]
    cols = 8
    rows = (num_channels // cols) + (1 if num_channels % cols != 0 else 0)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f"Channel {i + 1}")

    for i in range(num_channels, len(axes)):
        axes[i].axis('off')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Feature maps saved at {save_path}")
    plt.show()

    return feature_maps


def compute_feature_map_statistics(feature_maps):

    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.cpu().numpy()

    if len(feature_maps.shape) == 2:
        raise ValueError("Expected feature_maps of shape (num_channels, H, W), but got (H, W)")

    num_channels, H, W = feature_maps.shape
    flattened_maps = feature_maps.reshape(num_channels, -1)

    # Compute statistics
    mean_activation = np.mean(flattened_maps, axis=1)
    std_dev = np.std(flattened_maps, axis=1)
    max_activation = np.max(flattened_maps, axis=1)
    min_activation = np.min(flattened_maps, axis=1)
    sparsity = 100 * np.sum(np.abs(flattened_maps) < 1e-3, axis=1) / (H * W)
    avg_entropy = np.mean([entropy(np.abs(fm) + 1e-6) for fm in flattened_maps])
    avg_skewness = np.mean([skew(fm) for fm in flattened_maps])
    avg_kurtosis = np.mean([kurtosis(fm) for fm in flattened_maps])

    return {
        "Mean Activation": np.mean(mean_activation),
        "Std Dev": np.mean(std_dev),
        "Max Activation": np.mean(max_activation),
        "Min Activation": np.mean(min_activation),
        "Feature Map Sparsity (%)": np.mean(sparsity),
        "Entropy": avg_entropy,
        "Skewness": avg_skewness,
        "Kurtosis": avg_kurtosis
    }


def figure_to_numpy(fig):
    """Convert a Matplotlib figure to a NumPy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image.convert("L"))


def main():
    model8 = YOLO("yolov8n-pose.pt")
    model8.eval()

    image_path = 'data/images/006454781.jpg' #cy
    # image_path = 'data/images/000041029.jpg' #ch
    layer8 = 'model.22.cv4.2.2'
    # layer8 = 'model.22.dfl'

    feature_map_8 = extract_feature_maps(model8, image_path, layer8, "last_layer8")
    if feature_map_8 is not None:
        stats = compute_feature_map_statistics(feature_map_8)
        # feature_map_8.savefig(f'chef_{layer8}_yolo8_maps.png')
        print(stats)
        # feature_map_8.close()

    model11 = YOLO("yolo11n-pose.pt")
    model11.eval()

    layer11 = 'model.23.cv4.2.2'
    # layer11 = 'model.23.dfl'


    feature_map_11 = extract_feature_maps(model11, image_path, layer11, "last_layer11")
    if feature_map_11 is not None:
        stats = compute_feature_map_statistics(feature_map_11)
        # feature_map_11.savefig(f'chef_{layer11}_yolo11_maps.png')
        print(stats)
        # feature_map_11.close()

if __name__ == '__main__':
  main()