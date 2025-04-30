import torch
import copy, math, pickle, json, os
import bertviz, uuid
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import argparse
import os
import random
import urllib.parse

from PIL import Image, ImageDraw
from transformers.image_transforms import (
    convert_to_rgb,
    get_resize_output_image_size,
    resize,
    center_crop)
from transformers.image_utils import (
    infer_channel_dimension_format,
    to_numpy_array)
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import time
import torch.nn as nn
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from sklearn.cluster import DBSCAN
from collections import Counter
from dataclasses import dataclass

LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM

@dataclass
class ImagePrompt:
    image_url: str
    prompt: str
    prefix: str

@dataclass
class Entropy:
    cluster: int
    count_points: int
    average_strength: float

def normalize(vector):
    max_value = max(vector)
    min_value = min(vector)
    vector1 = [(x-min_value)/(max_value-min_value) for x in vector]
    vector2 = [x/sum(vector1) for x in vector1]
    return vector2


def transfer_output(model_output):
    all_pos_layer_input = []

    all_pos_layer_output = []
    all_last_attn_subvalues = []

    for layer_i in range(LAYER_NUM):
        cur_layer_input = model_output[layer_i][0]
        cur_layer_output = model_output[layer_i][4]
        cur_last_attn_subvalues = model_output[layer_i][5]

        all_pos_layer_input.append(cur_layer_input[0].tolist())

        all_pos_layer_output.append(cur_layer_output[0].tolist())
        all_last_attn_subvalues.append(cur_last_attn_subvalues[0].tolist())

    return all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues

def get_bsvalues(vector, model, final_var):
    vector = vector * torch.rsqrt(final_var + 1e-6)
    vector_rmsn = vector * model.language_model.model.norm.weight.data
    vector_bsvalues = model.language_model.lm_head(vector_rmsn).data
    return vector_bsvalues

def get_prob(vector):
    prob = torch.nn.Softmax(-1)(vector)
    return prob

def transfer_l(l):
    new_x, new_y = [], []
    for x in l:
        new_x.append(x[0])
        new_y.append(x[1])
    return new_x, new_y

def plt_bar(x, y, yname="log increase"):
    x_major_locator=MultipleLocator(1)
    plt.figure(figsize=(8, 3))
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt_x = [a/2 for a in x]
    plt.xlim(-0.5, plt_x[-1]+0.49)
    x_attn, y_attn, x_ffn, y_ffn = [], [], [], []
    for i in range(len(x)):
        if i%2 == 0:
            x_attn.append(x[i]/2)
            y_attn.append(y[i])
        else:
            x_ffn.append(x[i]/2)
            y_ffn.append(y[i])
    plt.bar(x_attn, y_attn, color="darksalmon", label="attention layers")
    plt.bar(x_ffn, y_ffn, color="lightseagreen", label="FFN layers")
    plt.xlabel("layer")
    plt.ylabel(yname)
    plt.legend()
    plt.show()

def plt_heatmap(data):
    xLabel = range(len(data[0]))
    yLabel = range(len(data))
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xticks(range(len(xLabel)))
    ax.set_xticklabels(xLabel)
    ax.set_yticks(range(len(yLabel)))
    ax.set_yticklabels(yLabel)
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    #plt.colorbar(im)
    plt.title("attn head log increase heatmap")
    plt.show()

        
class LlavaMechanism:
    def __init__(self, model_id="llava-hf/llava-1.5-7b-hf", device="cuda"):
        """
        Initialize the LlavaMechanism class by loading the model and processor.
        
        """
        torch.set_default_device('cuda')
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            revision='a272c74',
            output_attentions=True,
            output_hidden_states=True
        ).to(device)
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision='a272c74',
            do_resize=True,
            size={"height": 336, "width": 336},
            patch_size=14, 
            do_center_crop=True,
            do_normalize=True
        )
        
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)
    def get_attention_patches(self, images, prompts, prefixes):
        """
        Processes a true batch of different images with proper timing and validation
        
        Args:
            images (list): List of PIL.Image objects
            prompts (list): List of prompt strings
            prefixes (list): List of prefix strings
            
        Returns:
            list: List of tuples (demo_img, increase_scores_normalize)
        """
        
        if len(images) != len(prompts) or len(images) != len(prefixes):
            raise ValueError("Input lists must have equal length")
        if not images:
            return []
    
        batch_size = len(images)
        batch_results = []
        
        # Verify input is not duplicating
        input_hashes = [hash(img.tobytes()) for img in images]
        if len(set(input_hashes)) != batch_size:
            print(f"⚠️ Warning: Detected {batch_size - len(set(input_hashes))} duplicate inputs!")
    
        # Convert to RGB
        processed_images = []
        for img in images:
            try:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if min(img.size) < 336:
                    img = img.resize((336, 336))
                processed_images.append(img)
            except Exception as e:
                print(f"Image processing error: {e}")
                processed_images.append(Image.new('RGB', (336, 336)))
    
        try:
            # Processing
            inputs = self.processor(
                text=[f"USER: <image>\n{p}\nASSISTANT: {pre}" for p, pre in zip(prompts, prefixes)],
                images=processed_images,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
    
            # Foward pass with timing
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
            model_time = time.time() - start_time
            
            print(f"\n=== BATCH PROCESSING ===")
            print(f"Batch size: {batch_size}")
            print(f"Pure model time: {model_time:.2f}s")
            print(f"Time per image: {model_time/batch_size:.2f}s")
    
            # Process outputs
            first_attention = None
            for i in range(batch_size):
                try:
                    # Image processing
                    image_numpy = to_numpy_array(processed_images[i])
                    input_data_format = infer_channel_dimension_format(image_numpy)
                    output_size = get_resize_output_image_size(
                        image_numpy, size=336,
                        default_to_square=False,
                        input_data_format=input_data_format
                    )
                    image_resize = resize(
                        image_numpy, output_size,
                        resample=3,
                        input_data_format=input_data_format
                    )
                    demo_img = center_crop(
                        image_resize,
                        size=(336, 336),
                        input_data_format=input_data_format
                    )
    
                    # Attention processing
                    last_layer_attn = outputs.attentions[-1][i]  
                    visual_attention = last_layer_attn[:, -1, 5:581].mean(0)
                    attention_scores = visual_attention.detach().cpu().numpy()
                    
                    if i == 0:
                        first_attention = attention_scores.copy()
                    
                    if i > 0 and first_attention is not None:
                        diff = np.max(np.abs(attention_scores - first_attention))
                        if diff < 0.01:
                            print(f"⚠️ Warning: Result {i} shows minimal difference (max diff: {diff:.4f})")
    
                    increase_scores_normalize = normalize(attention_scores)
                    batch_results.append((demo_img, increase_scores_normalize))
    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    batch_results.append((np.zeros((336, 336, 3)), [0.0]*576))
    
        except Exception as e:
            print(f"Batch processing failed: {e}")
            batch_results = [(np.zeros((336, 336, 3)), [0.0]*576) for _ in range(batch_size)]
    
        return batch_results
        
    def save_vis(self, demo_img, increase_scores_normalize, output_path=None):
        """
        Save a visualization of the original image with overlayed attention patches.
        
        Args:
            demo_img (numpy.ndarray): The image to visualize
            increase_scores_normalize (list): Normalized attention scores
            output_path (str, optional): Path to save the visualization. If None, will use default.
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "attention_analysis.png")
        else:
            output_path = os.path.join(self.output_dir, f"{output_path}.png")
        demo_img_h, demo_img_w, demo_img_c = demo_img.shape
        
        # Resize attention scores
        demo_img_inc = np.array(increase_scores_normalize).reshape((24, 24))
        demo_img_inc = cv2.resize(demo_img_inc,
                                dsize=(demo_img_w, demo_img_h),
                                interpolation=cv2.INTER_CUBIC)
        
        # Create the visualization
        plt.figure(figsize=(25, 6))
        
        # Plot target image
        plt.subplot(1, 3, 1)
        plt.imshow(demo_img)
        plt.axis("off")
        plt.title("image")
        
        # Plot image
        plt.subplot(1, 3, 2)
        plt.imshow(demo_img)
        plt.imshow(demo_img_inc, alpha=0.8, cmap="gray")
        plt.axis("off")
        plt.title("log increase")
        
        plt.savefig(output_path)
        plt.close()
        print(f"Visualization saved to {output_path}")

def transform_matrix_to_3d_points(array_2d: np.ndarray):
    """Transforms a 2D numpy array to an array of (x, y, value) tuples, here (x, y) is the location of the value.

    Example:
        Input: [[0.3 1.7 2.5]
                [0.1 1.2 1.9]
               ]
        Output:
            [[0, 0, 0.3]
             [0, 1, 1.7]
             [0, 2, 2.5]
             [1, 0, 0.1]
             [1, 1, 1.2]
             [1, 2, 1.9]
            ]
    Args:
        array_2d: A 2D numpy array.

    Returns:
        A new numpy array where each element is a tuple (x, y, value).
    """    
    rows, cols = array_2d.shape    
    result = np.empty([rows * cols, 3], dtype=object)

    for x in range(rows):
        for y in range(cols):
            result[x * cols + y] = [y, -x + 23, array_2d[x, y]]

    return result

def find_clusters(attentions_with_locations: np.ndarray, eps: float, min_samples: int, metric: str="euclidean") -> (DBSCAN, int, int):
    """Find clusters from a given attention 3D points.

    Args:
        attentions_with_locations: a list of attentions with location info.
        eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        metric: the custom metric to calculate distances between instances in the provided feature array.

    Returns:
        The DBSCAN object, the number of clusters and the number of noise points.
    """
    # Get the coordinates of the patches.
    x_coords = attentions_with_locations[:, 0]
    y_coords = attentions_with_locations[:, 1]
    coords = np.stack((x_coords, y_coords), axis=-1)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    return db, n_clusters_, n_noise_

def apply_threshold(datapoints: np.ndarray, percentile: float) -> np.ndarray:
    """Remove the lowest percentile scores.
    """
    z_values = datapoints[:, 2]

    # Calculate the percentile
    p_value = np.percentile(z_values, percentile)
    print(f"{percentile}th percentile value: {p_value}")

    return datapoints[datapoints[:, 2] > p_value]

def duplicate_points(datapoints: np.ndarray, min_dup: int, max_dup: int) -> np.ndarray:
    """ Duplicate datapoints so that large valkues duplicates more times than smaller values.
    """
    # Extract value (attention score)
    values = datapoints[:, 2]
   
    # Normalize value to get number of times to duplicate
    scaled = ((values - values.min()) / (values.max() - values.min()) * (max_dup - min_dup) + min_dup + 1).astype(int)
   
    # Duplicate each point according to its scaled weight
    weighted_points = np.concatenate([np.repeat([pt], rep, axis=0) for pt, rep in zip(datapoints, scaled)], axis=0)
   
    # print(weighted_points)
    print(f"Original points: {len(datapoints)} -> After weighting: {len(weighted_points)}")
    return weighted_points

def save_attentions(weighted_attentions_with_locations: np.ndarray, db: DBSCAN, image_url: str):
    parsed_url = urllib.parse.urlparse(image_url)
    filename = os.path.basename(parsed_url.path)

    plt.scatter(weighted_attentions_with_locations[:, 0], weighted_attentions_with_locations[:, 1], c=db.labels_)
    plt.show()
    plt.savefig(os.path.join("output_images", "attention_analysis_" + filename))
    plt.close()

def calculate_entropy(weighted_attentions_with_locations: np.ndarray, db: DBSCAN):
    entropy = {}
    labels = db.labels_

    # Count the number of points per cluster
    cluster_counts = Counter(labels)
    
    # Count the points per cluster.
    for label, count in cluster_counts.items():
        if label == -1:
            print(f"Noise (unclustered): {count} points")
        else:
            entropy[label] = Entropy(label, count, 0.0)

    # Calculate the average strength per cluster.
    unique_clusters = set(labels) - {-1}  # Remove noise (-1)
    cluster_strengths = {}
    z_values = weighted_attentions_with_locations[:, 2]
    
    for cluster in unique_clusters:
        cluster_points = z_values[labels == cluster]  # Get strength values for the cluster
        cluster_strengths[cluster] = np.mean(cluster_points)
    
    for cluster, avg_strength in cluster_strengths.items():
        entropy[cluster].average_strength = avg_strength

    return entropy
        
def main():
    """
    Main function to demonstrate the usage of LlavaMechanism class.
    """
    mechanism = LlavaMechanism()
    
    test_image_url = "http://images.cocodataset.org/val2017/000000219578.jpg"
    test_prompt = "What is the color of the dog?"
    test_prefix = "the dog is the color"

    # Load images
    try:
        response = requests.get(test_image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Create batch
    batch_size = 5
    images = []
    prompts = []
    prefixes = []

    for i in range(batch_size):
        img = image.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle([i*20, i*20, i*20+50, i*20+50], fill=(i*50, 255-i*40, i*30))
        images.append(img)
        
        prompts.append(f"What is the color of pattern {i}?")
        prefixes.append(f"the pattern is color")

    # Process batch with timing
    start_time = time.time()
    batch_results = mechanism.get_attention_patches(images, prompts, prefixes)
    total_time = time.time() - start_time

    # Batch verification
    print("\n=== BATCH VERIFICATION ===")
    print(f"Processed {len(batch_results)} items in {total_time:.2f}s")
    
    input_hashes = [hash(img.tobytes()) for img in images]
    unique_inputs = len(set(input_hashes))
    print(f"Unique inputs: {unique_inputs}/{batch_size} {'✅' if unique_inputs == batch_size else '❌'}")

    first_scores = np.array(batch_results[0][1])
    diffs = [np.max(np.abs(np.array(r[1]) - first_scores)) for r in batch_results[1:]]
    avg_diff = np.mean(diffs)
    print(f"Avg output difference: {avg_diff:.4f} {'✅' if avg_diff > 0.01 else '❌ (possible duplicates)'}")

    # Verify results are identical 
    first_scores_np = np.array(batch_results[0][1])

    for i, (image, scores) in enumerate(batch_results):
        scores_np = np.array(scores)  
        
        if not np.array_equal(scores_np, first_scores_np):
            print(f"Score differences in result {i}:")
            print("Max diff:", np.max(np.abs(scores_np - first_scores_np)))
        elif not np.array_equal(image, batch_results[0][0]):
            print(f"Image differences in result {i}")
        else:
            print(f"Result {i} matches first result perfectly")

    # Process and visualize (only first result because they should be the same)
    demo_img, increase_scores_normalize = batch_results[0]
    
    # Save visualization
    mechanism.save_vis(demo_img, increase_scores_normalize, test_prompt)
    
    # Create attention matrix for visualization
    attention_matrix = np.array(increase_scores_normalize).reshape(24, 24)
    
    # Visualize the raw attention matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Strength')
    plt.title("Raw Attention Matrix (24x24)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(False)
    
    matrix_path = os.path.join("output_images", "attention_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"\nAttention matrix saved to {matrix_path}")
    
    # Cluster analysis
    attentions_with_locations = transform_matrix_to_3d_points(attention_matrix)
    print(f"Attentions with locations: {attentions_with_locations.shape}")
    
    # Remove lower percentile datapoints
    threshold_percentile = 60
    filtered_attentions_with_locations = apply_threshold(attentions_with_locations, threshold_percentile)
    print(f"Attentions without the lowest {threshold_percentile}% datapoints: {filtered_attentions_with_locations.shape}")
    
    # Duplicate datapoints
    weighted_attentions_with_locations = duplicate_points(filtered_attentions_with_locations, 1, 9)
    
    # Find clusters
    db, n_clusters, n_noise = find_clusters(weighted_attentions_with_locations, 1.3, 15)

    # Calculate entropy
    entropy = calculate_entropy(weighted_attentions_with_locations, db)
    print("\n=== CLUSTER RESULTS ===")
    print(f"Clusters: {n_clusters}, Noise points: {n_noise}")
    print("Cluster details:", entropy)
    
    # Save cluster visualization
    save_attentions(weighted_attentions_with_locations, db, test_image_url)

if __name__ == "__main__":
    main()
