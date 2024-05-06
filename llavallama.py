from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import requests
from PIL import Image
from jinja2 import Template

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)


# model_id = "llava-hf/llava-1.5-13b-hf"
model_id = "llava-hf/llava-1.5-7b-hf"
model_id = 'xtuner/llava-llama-3-8b-v1_1-transformers'

processor = AutoProcessor.from_pretrained(model_id, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")


print("GENERAL CASE ############################################################")
prompts = "USER: <image>\nWhat is shown in the image?\nASSISTANT:"
inputs = processor(prompts, images=[image1], return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)

# print("NO IMAGE CASE ############################################################")

# prompts_wo_image = "USER: What are you?\nASSISTANT:"
# inputs = processor.tokenizer(prompts_wo_image, return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)


# print("NoImage, image, no image reference CASE ############################################################")

# prompts_image_2 = "USER: What are you?\nASSISTANT:I am Vicuna, a virtual assistant. How can I help you?\nUSER: <image>\nWhat is shown in the image?\nASSISTANT: A lake, A deck and A forest.\nUSER\nTell me more about the image.\nASSISTANT:"
# inputs = processor(prompts_image_2, images=[image1], return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)


# print("NoImage, image, no image reference, new image CASE ############################################################")

# prompts_image_3 = "USER: What are you?\nASSISTANT:I am Vicuna, a virtual assistant. How can I help you?\nUSER: <image>\nWhat is shown in the image?\nASSISTANT: A lake, A deck and A forest.\nUSER\nTell me more about the image.\nASSISTANT:An amazing scenery providing a glancing view of a port or deck with a lake surrounded by a lush green forest.\nUSER: <image>\nWhat is shown in the image?\nASSISTANT:"
# inputs = processor(prompts_image_3, images=[image1, image2], return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)


# print("NoImage, image, no image reference, new image, no reference CASE ############################################################")

# prompts_image_3 = "USER: What are you?\nASSISTANT:I am Vicuna, a virtual assistant. How can I help you?\nUSER: <image>\nWhat is shown in the image?\nASSISTANT: A lake, A deck and A forest.\nUSER\nTell me more about the image.\nASSISTANT:An amazing scenery providing a glancing view of a port or deck with a lake surrounded by a lush green forest.\nUSER: <image>\nWhat is shown in the image?\nASSISTANT:Two cats laying on a couch.\nUSER:\nTell me more about the image?\nASSISTANT:"
# inputs = processor(prompts_image_3, images=[image1, image2], return_tensors="pt").to("cuda")

# output = model.generate(**inputs, max_new_tokens=200)
# generated_text = processor.batch_decode(output, skip_special_tokens=True)
# print(generated_text)

