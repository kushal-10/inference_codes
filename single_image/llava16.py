from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
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

image_padded = pad_images([image1, image2])

model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"
# model_id = "llava-hf/llava-v1.6-vicuna-13b-hf"
# model_id = "llava-hf/llava-v1.6-34b-hf"

vicuna_chat_template = '''A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.{% for message in messages %}{% if message['role'] == 'user' %}{% if message['image'] %}USER:<image>\n{{message['content']}}{% else %}USER:\n{{message['content']}}{% endif %}{% elif message['role'] == 'assistant' %}ASSISTANT:{{message['content']}}{% endif %}{% endfor %}ASSISTANT:'''
large_chat_template = "<|im_start|>system\nAnswer the questions.<|im_end|>{%- for message in messages -%}{% if message['role'] == 'user' %}{% if message['image']%}<|im_start|>user\n<image>\n{{message['content']}}<|im_end|>{% else %}<|im_start|>\nuser\n{{message['content']}}<|im_end|>{% endif %}{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{message['content']}}<|im_end|>{% endif %}{% endfor %}<|im_start|>assistant\n"
mistral_chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{% if message['image']%}[INST] <image>\n{{message['content']}} [/INST]{% else %}[INST]\n{{message['content']}} [/INST]{% endif %}{% elif message['role'] == 'assistant' %}{{message['content']}}{% endif %}{% endfor %}"

messages_single_image = [
    {'role': 'system', 'content': 'This is a system message'},
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': 'A lake, A deck and A forest.'},
    {'role': 'user', 'content': 'Explain a bit more about this image'}]

messages_multiple_image = [
    {'role': 'system', 'content': 'This is a system message'},
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': 'A lake, A deck and A forest.'},
    {'role': 'user', 'content': 'Explain a bit more about this image'},
    {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
    {'role': 'user', 'content': 'Explain a bit about this image.', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]

temp = Template(mistral_chat_template)
prompt_message1 = temp.render(messages=messages_single_image)
prompt_message2 = temp.render(messages=messages_multiple_image)

processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

inputs = processor(prompt_message1, images=image_padded, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print("Output for Single image")
print(generated_text)

inputs = processor(prompt_message2, images=image_padded, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print("Output for Multiple image")
print(generated_text)
