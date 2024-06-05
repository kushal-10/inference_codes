# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image1 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

question = 'What is in the image?'
msgs = [
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': ' A lake, A deck and A forest.'},
    {'role': 'user', 'content': 'Explain a bit more about this image'},
    {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
    {'role': 'user', 'content': 'What is shown in this image?', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]

res = model.chat(
    image=[image, image1],
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)

'''
Does not accept system message 
'''