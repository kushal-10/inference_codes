# import os
# print(os.getcwd())
# print(os.path.exists("../../../data/huggingface_cache"))

# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image
import requests

def pad_images(images):
    # Determine the maximum width and height among all images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    # Create and return a list of padded images
    padded_images = []
    for image in images:
        # Create a new image with a black background
        new_image = Image.new("RGB", (max_width, max_height))

        # Calculate the position to paste the image so that it's centered
        x = (max_width - image.size[0]) // 2
        y = (max_height - image.size[1]) // 2

        # Paste the original image onto the new image
        new_image.paste(image, (x, y))
        padded_images.append(new_image)

    return padded_images

# # TODO support fast tokenizer here
# processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf", use_fast=False, cache_dir="../../../data/huggingface_cache")

# model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, cache_dir="../../../data/huggingface_cache")
# # processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf", use_fast=False)

# model = AutoModelForVision2Seq.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16)
# model.to("cuda")

# prepare image and text prompt, using the appropriate prompt template
image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

im_list = [image1, image2]
im_list = pad_images(im_list)

from jinja2 import Template
messages = [
    {'role': 'user', 'content': 'Are there any clouds in the image? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/1.jpg'},
    {'role': 'assistant', 'content': 'Yes'},
    {'role': 'user', 'content': 'Are there any chickens in the picture? Answer with only "Yes" or "No".', 'image': 'games/cloudgame/resources/images/1.jpg'}
]

# "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nAre there any clouds in the image? Answer with only 'Yes' or 'No'. ASSISTANT:Yes\nUSER: <image>\nAre there any chickens in the picture? Answer with only "Yes" or "No".ASSISTANR: "

templ = "{%- for message in messages -%}{% if message['role'] == 'user' %}<|im_start|>user\n<image>\n<{{ message['content'] }}><|im_end|>{% else %} <|im_start|>assistant\n {{ message['content'] }}{% endif %}{% endfor %}<|im_start|>assistant\n"
mistral_templ = "{% for message in messages %}{% if message['role'] == 'user' %}[INST]  <image>\n{{ message['content']}} [/INST] {% else %}\n{{message['content']}}\n{% endif %}{% endfor %}"
vicuna_templ = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.{% for message in messages %}{% if message['role'] == 'user' %}\nUSER: <image>\n{{ message['content'] }}{% elif message['role'] == 'assistant' %}ASSISTANT {{ message['content'] }} {% endif %}{% endfor %} ASSISTANT:"

template = Template(vicuna_templ)
rendered_text = template.render(messages=messages)
print(rendered_text)

# inputs = processor(rendered_text, im_list, return_tensors="pt").to("cuda")

# # autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)

# print(processor.decode(output[0], skip_special_tokens=True))
