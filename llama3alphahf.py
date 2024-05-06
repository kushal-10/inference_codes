import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests


model_id = "qresearch/llama-3-vision-alpha-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)


print(
    tokenizer.decode(
        model.answer_question(image1, image2, "Compare these two images?", tokenizer),
        skip_special_tokens=True,
    )
)
