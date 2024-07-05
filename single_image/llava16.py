# from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
# import requests
# from PIL import Image
# from jinja2 import Template
#
#
# def load_image(image_path: str) -> Image:
#     '''
#     Load an image from a given link/directory
#
#     :param image_path: A string that defines the link/directory of the image
#     :return image: Loaded image
#     '''
#     if image_path.startswith('http') or image_path.startswith('https'):
#         image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
#     else:
#         image = Image.open(image_path).convert('RGB')
#     return image
#
#
#
# image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
# image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
#
# # images_padded1 = pad_images([image1])
# # images_padded2 = pad_images([image1, image2])
#
# # model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# # model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# # model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "llava-hf/llava-v1.6-34b-hf"
#
# vicuna_chat_template = '''A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.{% for message in messages %}{% if message['role'] == 'user' %}{% if message['image'] %}USER:<image>\n{{message['content']}}{% else %}USER:\n{{message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}ASSISTANT:{{message['content']}}{% endif %}{% endfor %}ASSISTANT:'''
# large_chat_template = "<|im_start|>system\nAnswer the questions.<|im_end|>{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image']%}<|im_start|>user\n<image>\n{{message['content']}}<|im_end|>{% else %}<|im_start|>\nuser\n{{message['content']}}<|im_end|>{% endif %}{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{message['content']}}<|im_end|>{% endif %}{% endfor %}<|im_start|>assistant\n"
# mistral_chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{% if message['image']%}[INST] <image>\n{{message['content']}} [/INST]{% else %}[INST]\n{{message['content']}} [/INST]{% endif %}{% elif message['role'] == 'assistant' %}{{message['content']}}{% endif %}{% endfor %}"
#
# messages_single_image = [
#     {'role': 'system', 'content': 'This is a system message'},
#     {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
#     {'role': 'assistant', 'content': 'A lake, A deck and A forest.'},
#     {'role': 'user', 'content': 'Explain a bit more about this image'}]
#
# messages_multiple_image = [
#     {'role': 'system', 'content': 'This is a system message'},
#     {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
#     {'role': 'assistant', 'content': 'A lake, A deck and A forest.'},
#     {'role': 'user', 'content': 'Explain a bit more about this image'},
#     {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
#     {'role': 'user', 'content': 'Explain a bit about this image.', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
# ]
#
# temp = Template(large_chat_template)
# prompt_message1 = temp.render(messages=messages_single_image)
# prompt_message2 = temp.render(messages=messages_multiple_image)
#
# processor = AutoProcessor.from_pretrained(model_id, device_map="auto", verbose=False)
# model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto", low_cpu_mem_usage=True)
# print(model)
#
# inputs = processor(prompt_message1, images=[image1], return_tensors="pt").to("cuda")
# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print("Output for Single image")
# print(generated_text)
#
# # inputs = processor(prompt_message2, images=images_padded2, return_tensors="pt").to("cuda")
# # output = model.generate(**inputs, max_new_tokens=200)
# # generated_text = processor.batch_decode(output, skip_special_tokens=True)
# # print("Output for Multiple image")
# # print(generated_text)

## Try base inference
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-34b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"

inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=100)

print(processor.decode(output[0], skip_special_tokens=True))
