from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from transformers.generation import GenerationConfig
import torch
# torch.manual_seed(1234)

import requests
from PIL import Image

# image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
# image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
image1 = "https://llava-vl.github.io/static/images/view.jpg"
image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"

#TODOs
'''
-> Check Remote Code true changes/ So far other inferences work  (DONE)
-> Check history, can shift it to messages format and pass as a single prompt? (DONE)
-> Check multiple images (DONE)
-> Integrate with backend, add args for model loading in model registry

Current Changes 
1) Vis2Seq -> CausalLM
2) Padding=False 
'''

'''
Base Tokenizer inference
'''

# Note: The default behavior now has injection attack prevention off.
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# # 1st dialogue turn
# query = tokenizer.from_list_format([
#     {'image': image1},
#     {'text': 'What is shown in the image?'},
# ])

# response, history = model.chat(tokenizer, query=query, history=None)
# print("Response: ")
# print(response)
# print("History: ")
# print(history)

# 2nd dialogue turn
# query2 = tokenizer.from_list_format([
#     {'image': image1}, # Sanity check
#     {'text': 'Is this same image as the previous image?'}
# ])

# # 2nd dialogue turn
# query2 = tokenizer.from_list_format([
#     {'image': image2},
#     {'text': 'Is this same image as the previous image?'}
# ])

# response, history = model.chat(tokenizer, query=query2, history=history)
# print("Response: ")
# print(response)
# print("History: ")
# print(history)

'''
Tokenizer inference + Multiple images
'''

# query_multi = tokenizer.from_list_format([
#     {'image': image1},
#     {'text': 'Are there any clouds in the image?'},
#     {'text': 'No'},
#     {'image': image2},
#     {'text': 'How do these two images differ?'}
# ])

# response, history = model.chat(tokenizer, query=query_multi, history=None)
# print("Response: ")
# print(response)
# print("History: ")
# print(history)


'''
Processor inference
'''

# processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()

# # USE THIS TEMPLATE - 
# # MERGE CLEAN MESSAE AND GET IMAGES, ONE IMAGE PER TURN + RETURN LIST
# prompt_text1 = '''
# Picture 1: <img>https://llava-vl.github.io/static/images/view.jpg</img>\nWhat is shown in the image?\nThe image shows a wooden pier extending into a lake.
# Picture 2: <img>http://images.cocodataset.org/val2017/000000039769.jpg</img>\nHow does this image differ than the first image?
# '''

# # From Paper
# prompt_text2 = '''
# <im_start>user
# Picture 1: <img>https://llava-vl.github.io/static/images/view.jpg</img>Are there any clouds in the image?<im_end>
# <im_start>assistant
# Yes.<im_end>
# <im_start>user
# Picture 2: <img>http://images.cocodataset.org/val2017/000000039769.jpg</img>How do these two images differ?<im_end>
# <im_start>assistant
# '''

# images = [image1, image2]

# # prompt_text1 is compatible here, gets the difference between two images
# inputs = processor(prompt_text1, images=images, padding=False, return_tensors="pt").to("cuda")
# model_output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(model_output, skip_special_tokens=True)

# print(generated_text)
# # print(generated_text[0].split('\n')[-1])
# # for text in generated_text:
#     # response_text = text.split()[-1] # Get the last assistant response


'''
Processor Inference + Jinja template (Backend compatible)
'''
processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True).eval()

messages = [
    {'role': 'user', 'content': 'Are there any clouds in the image? Answer with only "Yes" or "No".', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': 'Yes'},
    {'role': 'user', 'content': 'This seems correct. Are there any chickens in the picture? Answer with only "Yes" or "No".', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]

prompt_text2 = '''
<im_start>user
Picture 1: <img>https://llava-vl.github.io/static/images/view.jpg</img>Are there any clouds in the image? Answer with only "Yes" or "No".<im_end>
<im_start>assistant
Yes.<im_end>
<im_start>user
Picture 2: <img>http://images.cocodataset.org/val2017/000000039769.jpg</img>This seems correct. Are there any chickens in the picture? Answer with only "Yes" or "No".<im_end>
<im_start>assistant
'''


from jinja2 import Template

#Final chat template (from paper)
qwen_template_str = '''{% set image_count = 0 %}{% for message in messages %}{% if message.role == 'user' %}{% if message.image %}{% set image_count = image_count + 1 %}<im_start>user\nPicture {{ image_count }}: <img>{{ message.image }}</img>{{ message.content }}<im_end>\n{% else %}<im_start>user\n{{ message.content }}<im_end>\n{% endif %}{% elif message.role == 'assistant' %}<im_start>assistant\n{{ message.content }}<im_end>\n{% endif %}{% endfor %}<im_start>assistant'''
template_str = qwen_template_str

# Rendering the template
template = Template(template_str)
rendered_text = template.render(messages=messages)

print(rendered_text)

images = [image1, image2]

# prompt_text1 is compatible here, gets the difference between two images
inputs = processor(rendered_text, images=images, padding=False, return_tensors="pt").to("cuda")
model_output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(model_output, skip_special_tokens=True)

print(generated_text)
'''
Output
.......<im_start>assistant\nNo<im_end>\n<im_start>user..........
'''
