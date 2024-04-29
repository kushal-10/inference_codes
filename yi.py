from PIL import Image
import requests
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForVision2Seq, AutoProcessor


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
# image1 = "https://llava-vl.github.io/static/images/view.jpg"
# image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"



tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-VL-34B")
processor = AutoProcessor.from_pretrained("01-ai/Yi-VL-34B")

model = AutoModelForVision2Seq.from_pretrained("01-ai/Yi-VL-34B").to('cuda')


'''
With Processor
'''
prompt3 = '''
USER <image>
What can you see in the image? ASSISTANT: 
'''
inputs = processor(prompt3, padding=True, return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)