from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration
import requests
from PIL import Image
from jinja2 import Template

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
image3 = Image.open(requests.get("https://farm7.staticflickr.com/6116/6263367172_2e52beb0b5_z.jpg", stream=True).raw)

image_padded = pad_images([image2, image1, image3])

# model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
model_id = "llava-hf/llava-v1.6-34b-hf"

processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
#large_chat_template = "<|im_start|>system\nAnswer the questions.<|im_end|>{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image']%}<|im_start|>user\n<image>\n{{message['content']}}<|im_end|>{% else %}<|im_start|>\nuser\n{{message['content']}}<|im_end|>{% endif %}{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{message['content']}}<|im_end|>{% endif %}{% endfor %}<|im_start|>assistant\n"

# Updated prompt based on - https://github.com/haotian-liu/LLaVA/pull/432

prompt_large = '''<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nYou are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.\n<image>\nThe first image is a distractor, <image>\nthe second image is the target, and <image>\nthe third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag "Expression: " for the given target image. Omit any other text<|im_end|><|im_start|>assistant\n'''

inputs = processor(prompt_large, images=image_padded, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=512)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)

