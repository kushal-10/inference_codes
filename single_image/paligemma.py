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
image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

# Use a temp API token, remove after testing form HF account

model = AutoModelForVision2Seq.from_pretrained(model_id, token=huggingface_api_key).eval()
processor = AutoProcessor.from_pretrained(model_id, token=huggingface_api_key)
model_config = AutoConfig.from_pretrained(model_id, token=huggingface_api_key)

#8192 context

# Check chat template
# Doesn't require an <image> token
messages = [
  {"role": "user", "content": "How are these 3 images different than one another?"}
]

# Doesn't have a chat template, because not chat-optimized, try with llava template anyway :)
# Remove <image> token from template
jinja_template = "{%- for message in messages -%}{% if message['role'] == 'user' %}\nUSER:\n{{message['content']}}{% elif message['role'] == 'assistant' %}\nASSISTANT:{{message['content']}}{% endif %}{% endfor %}\nASSISTANT:"

temp = Template(jinja_template)
prompt = temp.render(messages=messages)

model_inputs = processor(text=prompt, images=[image, image1, image2], return_tensors="pt")
# input_len = model_inputs["input_ids"].shape[-1]

generation = model.generate(**model_inputs, max_new_tokens=100)
# generation = generation[0][input_len:]
print(f"generation - {generation}")
decoded = processor.batch_decode(generation, skip_special_tokens=True)
print(f"decoded - {decoded}")

split_prefix = "ASSISTANT:"
response_text = decoded[0].split(split_prefix)[-1] # Get the last assistant response
cull=False
if cull:
    rt_split = response_text.split(cull) # Cull from End of String token
    response_text = rt_split[0]
response_text = response_text.strip()
print(f"Response text - {response_text}")


'''
SOME RESPONSES - 
# Single image
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

Accepts two images (Doesnot throw an error, but bad response)
Q - "What can you see in this image? Explain in extreme detail"
A - "A green car in front of a wall."
Q - "What can you see in this image?"
A - Car.

Three images (image of a car in front of a wall, lake view, two cats on a couch)
Q - How are these 3 images different than one another?
A - Sorry, as a base VLM I am not trained to answer this question.
'''