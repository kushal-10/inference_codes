from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import requests
from PIL import Image
from jinja2 import Template

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)


# model_id = "llava-hf/llava-1.5-13b-hf"
# model_id = "llava-hf/llava-1.5-7b-hf"
model_id = 'xtuner/llava-llama-3-8b-v1_1-transformers'

# processor = AutoProcessor.from_pretrained(model_id, device_map="auto", verbose=False)
# model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

# Chat Template:
'''
"<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>
 <|start_header_id|>assistant<|end_header_id|>\n\n"
'''
mess = [
    {'role': 'system', 'content': 'This is a system message'},
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    # {'role': 'assistant', 'content': ' A lake, A deck and A forest.'},
    # {'role': 'user', 'content': 'Explain a bit more about this image'},
    # {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
    # {'role': 'user', 'content': 'What is shown in this image?', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]


jinja_template = '''{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image'] %}<|start_header_id|>user<|end_header_id|>\n\n<image>\n{{message['content']}}<|eot_id|>{% else %}<|start_header_id|>user<|end_header_id|>\n\n{{message['content']}}<|eot_id|>{% endif %}{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>\n\n{{message['content']}}<|eot_id|>{% endif %}{% endfor %}<|start_header_id|>assistant<|end_header_id|>\n\n'''

temp = Template(jinja_template)
prompts = temp.render(messages=mess)
print(prompts)

# Passes all cases (with/without/ref images ....)

inputs = processor(prompts, images=[image1], return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
