from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from PIL import Image
import requests
import torch
from jinja2 import Template
import json


# Load the JSON data from the file
with open('key.json', 'r') as file:
    data = json.load(file)

# Retrieve the api_key for huggingface
huggingface_api_key = data['huggingface']['api_key']


model_id = "google/paligemma-3b-mix-224"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

# Use a temp API token, remove after testing form HF account

model = AutoModelForVision2Seq.from_pretrained(model_id, token=huggingface_api_key).eval()
processor = AutoProcessor.from_pretrained(model_id, token=huggingface_api_key)
model_config = AutoConfig.from_pretrained(model_id, token=huggingface_api_key)

#8192 context

# Check chat template
messages = [
  {"role": "user", "content": "What can you see in this image? Explain in extreme detail", 'image': 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true'},
  {"role": "assistant", "content": "A green car in front of a wall."},
  {"role": "user", "content": "Explain more about this image"},
]

# Doesn't have a chat template, because not chat-optimized, try with llava template anyway :)
jinja_template = "{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image'] %}\nUSER: \n{{message['content']}}{% else %}\nUSER:\n{{message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}\nASSISTANT:{{message['content']}}{% endif %}{% endfor %}\nASSISTANT:"

temp = Template(jinja_template)
prompt = temp.render(messages=messages)

# # Instruct the model to create a caption in Spanish
# prompt = "caption es"
model_inputs = processor(text=prompt, images=image, return_tensors="pt")
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)


'''
SOME RESPONSES - 

Q - What can you see in this image? Explain in extreme detail
A - Sorry, as a base VLM I am not trained to answer this question.

Q - "What can you see in this image? Explain in extreme detail"
A - "A green car in front of a wall."
Q - "Explain more about this image"
A - Car is parked on the road.
Car is green in color.
Car is on the road.
Car is parked on the road.
Car is on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
Car is parked on the road.
'''