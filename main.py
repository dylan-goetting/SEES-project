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

from PIL import Image
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

LAYER_NUM = 32
HEAD_NUM = 32
HEAD_DIM = 128
HIDDEN_DIM = HEAD_NUM * HEAD_DIM

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
        
        Args:
            model_id (str): The model ID to load
            device (str): Device to run the model on
        """
        # Setup CUDA
        torch.set_default_device('cuda')
        
        # Load model
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            low_cpu_mem_usage=True, 
            revision='a272c74',
        ).to(device)
        
        print(f"Model loaded on {self.model.device}")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')
        
        # Create output directory for saved visualizations
        self.output_dir = "output_images"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_attention_patches(self, image, prompt, prefix):
        """
        Get attention patches for an image with a specific prompt.
        
        Args:
            image (PIL.Image): Input image
            prompt (str): The prompt text
            prefix (str): The prefix text after ASSISTANT tag
            
        Returns:
            tuple: (demo_img, increase_scores_normalize)
        """
        t = time.time()
        
        # Process input
        full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: {prefix}"
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        print(f'Finished inference time {time.time() - t}')
        
        # Get output probabilities
        outputs_probs = get_prob(outputs["logits"][0][-1])
        outputs_probs_sort = torch.argsort(outputs_probs, descending=True)
        print([self.processor.decode(x) for x in outputs_probs_sort[:10]])
        print(outputs_probs_sort[:10].tolist())
        
        # Process model outputs
        all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues = transfer_output(outputs[2])
        print(f'Finished transfer output time {time.time() - t}')
        final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)
        
        # Process image
        resample = 3
        crop_size = {"height": 336, "width": 336}
        image_convert = convert_to_rgb(image)
        image_numpy = to_numpy_array(image_convert)
        input_data_format = infer_channel_dimension_format(image_numpy)
        output_size = get_resize_output_image_size(image_numpy, size=336,
                     default_to_square=False, input_data_format=input_data_format)
        image_resize = resize(image_numpy, output_size, resample=resample, input_data_format=input_data_format)
        image_center_crop = center_crop(image_resize, size=(crop_size["height"], crop_size["width"]), input_data_format=input_data_format)
        
        demo_img = image_center_crop
        
        predict_index = outputs_probs_sort[0].item()
        print(predict_index, self.processor.decode(predict_index))
        
        # Calculate head-level increase
        all_head_increase = []
        for test_layer in range(LAYER_NUM):
            cur_layer_input = torch.tensor(all_pos_layer_input[test_layer])
            cur_v_heads = torch.tensor(all_last_attn_subvalues[test_layer])
            cur_attn_o_split = self.model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
            cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
            cur_attn_subvalues_head_sum = torch.sum(cur_attn_subvalues_headrecompute, 0)
            cur_layer_input_last = cur_layer_input[-1]
            origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, self.model, final_var))[predict_index])
            cur_attn_subvalues_head_plus = cur_attn_subvalues_head_sum + cur_layer_input_last
            cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
                    cur_attn_subvalues_head_plus, self.model, final_var))[:, predict_index])
            cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
            for i in range(len(cur_attn_plus_probs_increase)):
                all_head_increase.append([str(test_layer)+"_"+str(i), round(cur_attn_plus_probs_increase[i].item(), 4)])
        print(f'Finished head-level increase time {time.time() - t}')
        
        all_head_increase_sort = sorted(all_head_increase, key=lambda x:x[-1])[::-1]
        
        # Get the top head and calculate position increase
        test_layer, head_index = all_head_increase_sort[0][0].split("_")
        test_layer, head_index = int(test_layer), int(head_index)
        cur_layer_input = outputs[2][test_layer][0][0]
        cur_v_heads = outputs[2][test_layer][5][0]
        cur_attn_o_split = self.model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
        cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
        cur_attn_subvalues_headrecompute_curhead = cur_attn_subvalues_headrecompute[:, head_index, :]
        cur_layer_input_last = cur_layer_input[-1]
        origin_prob = torch.log(get_prob(get_bsvalues(
            cur_layer_input_last, self.model, final_var))[predict_index])
        cur_attn_subvalues_headrecompute_curhead_plus = cur_attn_subvalues_headrecompute_curhead + cur_layer_input_last
        cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
            cur_attn_subvalues_headrecompute_curhead_plus, self.model, final_var))[:, predict_index])
        cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
        head_pos_increase = cur_attn_plus_probs_increase.tolist()
        curhead_increase_scores = head_pos_increase[5:581]
        increase_scores_normalize = normalize(curhead_increase_scores)
        print(f'Finished getting patches time {time.time() - t}')
        
        return demo_img, increase_scores_normalize
    
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
        
        # Reshape and resize the attention scores
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
        
        # Plot image with overlay
        plt.subplot(1, 3, 2)
        plt.imshow(demo_img)
        plt.imshow(demo_img_inc, alpha=0.8, cmap="gray")
        plt.axis("off")
        plt.title("log increase")
        
        # Save the figure
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
    
    # Create LlavaMechanism instance
    mechanism = LlavaMechanism()

    imagePrompts = [ImagePrompt("http://images.cocodataset.org/val2017/000000219578.jpg", "What is the color of the dog?", "The color of the dog is")]

    for i, imagePrompt in enumerate(imagePrompts):
        print(f"\nProcess image {i} - {imagePrompt.image_url}")
        image = Image.open(requests.get(image_url, stream=True).raw)
        
        # Get attention patches
        demo_img, increase_scores_normalize = mechanism.get_attention_patches(image, prompt, prefix)
        
        # Save visualization
        mechanism.save_vis(demo_img, increase_scores_normalize, prompt)
        
        # increase_scores_normalize - min: 0.0, max: 0.1541638498880645
        # For each attention, prefix the patch row and column indices.
        increase_scores_normalize = np.array(increase_scores_normalize)
        increase_scores_normalize = increase_scores_normalize.reshape(24, 24)
    
        attentions_with_locations = transform_matrix_to_3d_points(increase_scores_normalize)
        print(f"Attentions with locations: ", attentions_with_locations.shape)
        
        # Remove lower percentile datapoints.
        threshold_percentile = 80
        filtered_attentions_with_locations = apply_threshold(attentions_with_locations, threshold_percentile)
        print(f"Attentions without the lowest {threshold_percentile}% datapoints: ", filtered_attentions_with_locations.shape)
        
        # Duplicate datapoints.
        weighted_attentions_with_locations = duplicate_points(filtered_attentions_with_locations, 1, 9)
        
        # Apply Euclidean distance to evaluate spatial proximity
        # epsilon = 1.5 - eps should be >=1 since the minimum distance between 2 adjacent attentions is 1.
        # min_samples = 15
        db, _, _ = find_clusters(weighted_attentions_with_locations, 1.3, 15)
    
        save_attentions(weighted_attentions_with_locations, db, image_url)
    
        # Calculate entropy.
        entropy = calculate_entropy(weighted_attentions_with_locations, db)
        print(entropy)
    
if __name__ == "__main__":
    main()
