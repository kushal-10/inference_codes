from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from PIL import Image
import requests
import torch
from jinja2 import Template
import json

#
# model_id = "llava-hf/llava-1.5-13b-hf"
# model = AutoModelForVision2Seq.from_pretrained(model_id, cache_dir="/project/kkoshti/hf_models")
# print(model)

model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir="/data/huggingface")