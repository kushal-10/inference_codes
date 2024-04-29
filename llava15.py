from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import requests
from PIL import Image
from jinja2 import Template

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)


# temp_message = ['USER:  <image> \nWhat is shown in the image?\nASSISTANT:  A lake, A deck and A forest.USER: \nExplain a bit more about this image\nASSISTANT: The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.USER:  <image> \nWhat is shown in this image?\nASSISTANT: The image shows a couch with two cats lying on it. One cat is sleeping, while the other cat is awake and looking at the camera. The couch is situated near a lake, creating a serene and peaceful atmosphere.']
# # Split the string
# message_parts = temp_message[0].split("ASSISTANT: ")
# print([message_parts[-1]])


mess = [
    {'role': 'system', 'content': 'This is a system message'},
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': ' A lake, A deck and A forest.'},
    {'role': 'user', 'content': 'Explain a bit more about this image'},
    {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
    {'role': 'user', 'content': 'What is shown in this image?', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]


# jinja_template_backup = '''{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image'] %}USER: <image>\n{{ message['content'] }}{% else %}USER: \n{{ message['content'] }}{% endif %}{% elif message['role'] == 'assistant' %}\nASSISTANT: {{ message['content'] }}{% endif %}{% endfor %}\nASSISTANT:'''
jinja_template = '''{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image'] %}USER:\n{{ message['content'] }}{% else %}USER: \n{{ message['content'] }}{% endif %}{% elif message['role'] == 'assistant' %}\nASSISTANT: {{ message['content'] }}{% endif %}{% endfor %}\nASSISTANT:'''

listt = []
temp = Template(jinja_template)
promptu = temp.render(messages=mess)
listt.append(promptu)
# model_id = "llava-hf/llava-1.5-13b-hf"
model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")


# prompt1 = '''USER <image>\nWhat is in the image?\nASSISTANT: A lake, A deck and A forest.\nUSER\nTell me more about the image.\nASSISTANT:'''
# prompts = prompt1
# prompts = listt[0]

# prompts = "USER: <image>\nWhat is shown in the image?\nASSISTANT:"
# inputs = processor(prompts, images=[image1], return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)

# ['USER  <image> \nWhat is in the image?\nASSISTANT: A lake, A deck and A forest.\nUSER\nTell me more about the image.\nASSISTANT: The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.']