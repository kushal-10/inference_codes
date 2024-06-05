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
    {'role': 'user', 'content': 'How are these two images different?', 'image': ['https://llava-vl.github.io/static/images/view.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg']}
]

res = model.chat(
    image=[image, image1],
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.0001,
    # system_prompt='' # pass system_prompt if needed
)
print(res)

'''
Does not accept system message 

Accepts multiple images but bad response

Doesnot work when temperature is strictly 0
'''