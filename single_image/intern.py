
# Requires a new package - decord==0.6.0

import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
image = ['./examples/cars1.jpg',
        './examples/cars2.jpg',
        './examples/cars3.jpg',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
print(f"First history {his}")
print(type(his[0]))

query = 'Image4 <ImageHere>; How about the car in Image4'
image.append('./examples/cars4.jpg')
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his2 = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, history=his, use_meta=True)
print(response)
print(f"Second history {his2}")

# query = 'Image5 <ImageHere>; How about the car in Image5'
# image.append('./examples/cars1.jpg')
# with torch.autocast(device_type='cuda', dtype=torch.float16):
#     response, his3 = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, history= his2, use_meta=True)
# print(response)
# print(f"Second history {his3}")



