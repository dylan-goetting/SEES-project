# Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering 

## Introduction

This repo is for the arxiv paper Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering 

This work explores the mechanism of Llava in visual question answering. 

This work uses the findings and insights in [this repo](https://github.com/zepingyu0512/neuron-attribution/tree/main) in [this EMNLP 2024 paper](https://zepingyu0512.github.io/neuron-attribution.github.io/), and [this repo](https://github.com/zepingyu0512/in-context-mechanism) in [this EMNLP 2024 paper](https://zepingyu0512.github.io/in-context-mechanism.github.io/)

## Running code

You can directly have a look at Llava_visualize.ipynb without running the code.

Environment versions: please see **environment.yml**

First, please use modeling_llava.py to replace the original file in the transformers path, which is usually in anaconda3/envs/YOUR_ENV_NAME/lib/python3.8/site-packages/transformers/models/llava. This modified file is useful for extracting the internal vectors during inference time. **Please remember to save the original file.** 

Then run Llava_visualize.ipynb to analyze the important image patches for the generations in Llava.

## cite us: 

```
@article{yu2024understanding,
  title={Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering},
  author={Yu, Zeping and Ananiadou, Sophia},
  journal={arXiv preprint arXiv},
  year={2024}
}
```


