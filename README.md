# Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering 

## Introduction

This repo is for the arxiv paper: [Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering](https://arxiv.org/pdf/2411.10950) 

This work explores the mechanism of Llava in visual question answering. The overall mechanism is an ICL mechanism similar to TextualQA. 

This work uses the techniques and insights in:

[EMNLP 2024: Neuron-Level Knowledge Attribution in Large Language Models](https://zepingyu0512.github.io/neuron-attribution.github.io/)

[EMNLP 2024: How do Large Language Models Learn In-Context? Query and Key Matrices of In-Context Heads are Two Towers for Metric Learning](https://zepingyu0512.github.io/in-context-mechanism.github.io/)

[EMNLP 2024: Interpreting Arithmetic Mechanism in Large Language Models through Comparative Neuron Analysis](https://zepingyu0512.github.io/arithmetic-mechanism.github.io/)

## Running code

You can have a look at the example in Llava_visualize_code.ipynb without running the code.

Environment versions: please see **environment.yml**

First, please use modeling_llava.py to replace the original file in the transformers path, which is usually in anaconda3/envs/YOUR_ENV_NAME/lib/python3.8/site-packages/transformers/models/llava. This modified file is useful for extracting the internal vectors during inference time. **Please remember to save the original file.** 

Then run Llava_visualize_code.ipynb to visualize the important image patches for the generations in Llava.

You can also use Llava_visualize_Gradio.ipynb to visualize the images case-by-case in an interactive tool using Gradio. Please use similar patterns to fit Llava: USER: <image>\nWhat is the color of the dog?\nASSISTANT: The color of the dog is

## cite us: 

wait for the arxiv file

