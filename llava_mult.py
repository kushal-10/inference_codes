from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import requests
from PIL import Image
from jinja2 import Template

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

'''
Test LLAVA 1.5
'''
print("TEST LLAVA 1.5")
model_id = "llava-hf/llava-1.5-13b-hf"
# model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

prompts = "USER: <image>\n<image>\nDescribe these two images in extreme detail\nASSISTANT:"
inputs = processor(prompts, images=[image1, image2], return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=512)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)

'''
TEST LLAVA 1.6
'''
print("TEST LLAVA 1.6")

def load_image(image_path: str) -> Image:
    '''
    Load an image from a given link/directory

    :param image_path: A string that defines the link/directory of the image 
    :return image: Loaded image
    '''
    if image_path.startswith('http') or image_path.startswith('https'):
        image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

def pad_images(images):
    '''
    Pad the images. Only used for LLaVA NeXT models
    Will be deprecated when issue https://github.com/huggingface/transformers/issues/29832 is closed
    '''
    # Determine the maximum width and height among all images
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    # Create and return a list of padded images
    padded_images = []
    for image in images:
        # Create a new image with a black background
        new_image = Image.new("RGB", (max_width, max_height))

        # Calculate the position to paste the image so that it's centered
        x = (max_width - image.size[0]) // 2
        y = (max_height - image.size[1]) // 2

        # Paste the original image onto the new image
        new_image.paste(image, (x, y))
        padded_images.append(new_image)

    return padded_images


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

image_padded = pad_images([image1, image2])

# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
model_id = "llava-hf/llava-v1.6-34b-hf"


processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
prompts = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n<image>\nDescribe what is shown in these two images in detail<|im_end|><|im_start|>assistant\n"


inputs = processor(prompts, images=image_padded, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=512)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)

