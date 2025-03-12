import torch
import copy, math, pickle, json, os
import bertviz, uuid
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
import argparse
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
from matplotlib.pyplot import MultipleLocator
from collections import Counter
from utils import *
import time

def main(args):
    # Constants
    LAYER_NUM = 32
    HEAD_NUM = 32
    HEAD_DIM = 128
    HIDDEN_DIM = HEAD_NUM * HEAD_DIM
    OUTPUT_DIR = "output_images"  # Directory to save images
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup CUDA
    torch.set_default_device("cuda")
    # torch.cuda.set_device(args.gpu)
    zero_tensor = torch.tensor([0.0]*4096)
    # Setup
    model_id = "llava-hf/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        low_cpu_mem_usage=True, 
        revision='a272c74',
    ).cuda()
    print(model.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, revision='a272c74')

    image_url = "http://images.cocodataset.org/val2017/000000219578.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)

    prompt = "USER: <image>\nWhat is the color of the dog?\nASSISTANT: The color of the dog is"
    t = time.time()
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model(**inputs)
    print(f'Finished inference time {time.time() - t}')
    outputs_probs = get_prob(outputs["logits"][0][-1])
    outputs_probs_sort = torch.argsort(outputs_probs, descending=True)
    print([processor.decode(x) for x in outputs_probs_sort[:10]])
    print(outputs_probs_sort[:10].tolist())
    all_pos_layer_input, all_pos_layer_output, all_last_attn_subvalues, = transfer_output(outputs[2])
    print(f'Finished transfer output time {time.time() - t}')
    final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True)

    resample = 3
    shortest_edge = 336
    crop_size = {"height": 336, "width": 336}
    image_convert = convert_to_rgb(image)
    image_numpy = to_numpy_array(image_convert)
    input_data_format = infer_channel_dimension_format(image_numpy)
    output_size = get_resize_output_image_size(image_numpy, size=336,
                default_to_square=False, input_data_format=input_data_format)
    image_resize = resize(image_numpy, output_size, resample=resample, input_data_format=input_data_format)
    image_center_crop = center_crop(image_resize, size=(crop_size["height"], crop_size["width"]), input_data_format=input_data_format)

    # print(image_numpy.shape, image_resize.shape, image_center_crop.shape)
    # img1 = Image.fromarray(image_numpy)
    # img2 = Image.fromarray(image_resize)
    demo_img = image_center_crop

    predict_index = outputs_probs_sort[0].item()
    print(predict_index, processor.decode(predict_index))

    #head-level increase
    all_head_increase = []
    for test_layer in range(LAYER_NUM):
        cur_layer_input = torch.tensor(all_pos_layer_input[test_layer])
        cur_v_heads = torch.tensor(all_last_attn_subvalues[test_layer])
        cur_attn_o_split = model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
        cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
        cur_attn_subvalues_head_sum = torch.sum(cur_attn_subvalues_headrecompute, 0)
        cur_layer_input_last = cur_layer_input[-1]
        origin_prob = torch.log(get_prob(get_bsvalues(cur_layer_input_last, model, final_var))[predict_index])
        cur_attn_subvalues_head_plus = cur_attn_subvalues_head_sum + cur_layer_input_last
        cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
                cur_attn_subvalues_head_plus, model, final_var))[:, predict_index])
        cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
        for i in range(len(cur_attn_plus_probs_increase)):
            all_head_increase.append([str(test_layer)+"_"+str(i), round(cur_attn_plus_probs_increase[i].item(), 4)])
    print(f'Finished head-level increase time {time.time() - t}')

    all_head_increase_sort = sorted(all_head_increase, key=lambda x:x[-1])[::-1]
    # print(all_head_increase_sort[:30])
    # all_head_increase_list = [x[1] for x in all_head_increase]
    # all_head_increase_list_split = torch.tensor(all_head_increase_list).view((LAYER_NUM, HEAD_NUM)).permute((1,0)).tolist()
    # plt.figure(figsize=(10, 8))
    # plt_heatmap(all_head_increase_list_split)
    # plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"))
    # plt.close()

    #pos increase
    pos_len = len(all_pos_layer_input[0])
    test_layer, head_index = 22, 27
    cur_layer_input = torch.tensor(all_pos_layer_input[test_layer])
    cur_v_heads = torch.tensor(all_last_attn_subvalues[test_layer])
    cur_attn_o_split = model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
    cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
    cur_attn_subvalues_headrecompute_curhead = cur_attn_subvalues_headrecompute[:, head_index, :]
    cur_layer_input_last = cur_layer_input[-1]
    origin_prob = torch.log(get_prob(get_bsvalues(
        cur_layer_input_last, model, final_var))[predict_index])
    cur_attn_subvalues_headrecompute_curhead_plus = cur_attn_subvalues_headrecompute_curhead + cur_layer_input_last
    cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
        cur_attn_subvalues_headrecompute_curhead_plus, model, final_var))[:, predict_index])
    cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
    print(f'Finished pos increase time {time.time() - t}')

    # head_pos_increase = cur_attn_plus_probs_increase.tolist()
    # head_pos_increase_zip = list(zip(range(pos_len), head_pos_increase))
    # head_pos_increase_zip_sort = sorted(head_pos_increase_zip, key=lambda x: x[-1])[::-1]
    # pos_score_sum = sum([x[1] for x in head_pos_increase_zip_sort])

    # cur_attn_plus_probs_increase_increase_zip = list(zip(range(len(cur_attn_plus_probs_increase)), 
    #     cur_attn_plus_probs_increase.tolist()))
    # cur_attn_plus_probs_increase_increase_zip_sort = sorted(cur_attn_plus_probs_increase_increase_zip,
    #     key=lambda x:x[-1])[::-1]
    # cur_layer_input_bsvalues = get_bsvalues(cur_layer_input, model, final_var)
    # cur_layer_input_bsvalues_sort = torch.argsort(cur_layer_input_bsvalues, descending=True)
    # cur_attn_subvalues_headrecompute_curhead_bsvalues = get_bsvalues(
    #     cur_attn_subvalues_headrecompute_curhead, model, final_var)
    # cur_attn_subvalues_headrecompute_curhead_bsvalues_sort = torch.argsort(
    #     cur_attn_subvalues_headrecompute_curhead_bsvalues, descending=True)
    # key_input = cur_layer_input.clone()
    # key_input -= torch.tensor(all_pos_layer_input[0])
    # for layer_i in range(test_layer):
    #     key_input -= torch.tensor(all_pos_ffn_output[layer_i])
    # key_input_bsvalues = get_bsvalues(key_input, model, final_var)
    # key_input_bsvalues_sort = torch.argsort(key_input_bsvalues, descending=True)
    # value_input = cur_layer_input.clone()
    # value_input -= torch.tensor(all_pos_layer_input[0])
    # for layer_i in range(test_layer):
    #     value_input -= torch.tensor(all_pos_attn_output[layer_i])
    # value_input_bsvalues = get_bsvalues(value_input, model, final_var)
    # value_input_bsvalues_sort = torch.argsort(value_input_bsvalues, descending=True)

    # for pos, increase in cur_attn_plus_probs_increase_increase_zip_sort[:10]:
    #     print("\n", pos, "increase: ", round(increase, 4), "attn: ", round(
    #         all_attn_scores[test_layer][0][head_index][-1][pos].item(), 4))
    #     print("layer input: ", [processor.decode(x) for x in cur_layer_input_bsvalues_sort[pos][:20]])
    #     print("key input: ", [processor.decode(x) for x in key_input_bsvalues_sort[pos][:20]])
    #     print("value input: ", [processor.decode(x) for x in value_input_bsvalues_sort[pos][:20]])
    #     print("value ov: ", [processor.decode(x) for x in cur_attn_subvalues_headrecompute_curhead_bsvalues_sort[pos][:10]])

    test_layer, head_index = all_head_increase_sort[0][0].split("_")
    test_layer, head_index = int(test_layer), int(head_index)
    cur_layer_input = outputs[2][test_layer][0][0]
    cur_v_heads = outputs[2][test_layer][5][0]
    cur_attn_o_split = model.language_model.model.layers[test_layer].self_attn.o_proj.weight.data.T.view(HEAD_NUM, HEAD_DIM, -1)
    cur_attn_subvalues_headrecompute = torch.bmm(cur_v_heads, cur_attn_o_split).permute(1, 0, 2)
    cur_attn_subvalues_headrecompute_curhead = cur_attn_subvalues_headrecompute[:, head_index, :]
    cur_layer_input_last = cur_layer_input[-1]
    origin_prob = torch.log(get_prob(get_bsvalues(
        cur_layer_input_last, model, final_var))[predict_index])
    cur_attn_subvalues_headrecompute_curhead_plus = cur_attn_subvalues_headrecompute_curhead + cur_layer_input_last
    cur_attn_plus_probs = torch.log(get_prob(get_bsvalues(
        cur_attn_subvalues_headrecompute_curhead_plus, model, final_var))[:, predict_index])
    cur_attn_plus_probs_increase = cur_attn_plus_probs - origin_prob
    head_pos_increase = cur_attn_plus_probs_increase.tolist()
    curhead_increase_scores = head_pos_increase[5:581]
    increase_scores_normalize = normalize(curhead_increase_scores)
    print(f'Finished getting patches time {time.time() - t}')


    # attn_scores_all = torch.tensor([0.0]*576)
    # for layer_index in range(LAYER_NUM):
    #     for head_index in range(HEAD_NUM):
    #         attn_scores = outputs[2][layer_index][7][0][head_index][-1][5:581]
    #         attn_scores_all += attn_scores
    # attn_scores_all = attn_scores_all/1024.0


    demo_img_h, demo_img_w, demo_img_c = demo_img.shape
    # demo_img_att = np.array(attn_scores_all.tolist()).reshape((24, 24))
    # demo_img_att = cv2.resize(demo_img_att,
    #                         dsize=(demo_img_w, demo_img_h),
    #                         interpolation=cv2.INTER_CUBIC)
    demo_img_inc = np.array(increase_scores_normalize).reshape((24, 24))
    demo_img_inc = cv2.resize(demo_img_inc,
                            dsize=(demo_img_w, demo_img_h),
                            interpolation=cv2.INTER_CUBIC)


    # plot with matplotlib
    plt.figure(figsize=(25, 6))

    # plot target image
    plt.subplot(1, 3, 1)
    plt.imshow(demo_img)
    plt.axis("off")
    plt.title("image")

    # # plot image with attention masked on it
    # plt.subplot(1, 3, 2)
    # plt.imshow(demo_img)
    # plt.imshow(demo_img_att, alpha=0.8, cmap="gray")
    # plt.axis("off")
    # plt.title("attention")

    # plot image with attention masked on it
    plt.subplot(1, 3, 2)
    plt.imshow(demo_img)
    plt.imshow(demo_img_inc, alpha=0.8, cmap="gray")
    plt.axis("off")
    plt.title("log increase")
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "attention_analysis.png"))
    plt.close()
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LLaVA analysis with specified GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    args = parser.parse_args()
    main(args)
