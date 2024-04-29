import json
import os
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
#flash_attn, peft

path = "OpenGVLab/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True).eval().cuda()
# Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     device_map='auto').eval()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
# set the max number of tiles in `max_num`
pixel_values = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()

generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# single-round single-image conversation
question = "请详细描述图片"
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(question, response)

# # multi-round single-image conversation
# question = "请详细描述图片"
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)

# question = "请根据图片写一首诗"
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)

# # multi-round multi-image conversation
# pixel_values1 = load_image('./examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = "详细描述这两张图片"
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(question, response)
# # 第一张图片是一只红熊猫，它有着独特的橙红色皮毛，脸部、耳朵和四肢的末端有白色斑块。红熊猫的眼睛周围有深色的环，它的耳朵是圆形的，上面有白色的毛。它正坐在一个木制的结构上，看起来像是一个平台或休息的地方。背景中有树木和竹子，这表明红熊猫可能在一个模拟自然环境的动物园或保护区内。
# #
# # 第二张图片是一只大熊猫，它是中国的国宝，以其黑白相间的皮毛而闻名。大熊猫的眼睛、耳朵和四肢的末端是黑色的，而它的脸部、耳朵内侧和身体其他部分是白色的。大熊猫正坐在地上，周围有竹子，这是它们的主要食物来源。背景中也有树木，这表明大熊猫可能在一个为它们提供自然栖息地模拟的动物园或保护区内。

# question = "这两张图片的相同点和区别分别是什么"
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(question, response)
# # 这两张图片的相同点：
# # 
# # 1. 都展示了熊猫，这是两种不同的熊猫物种。
# # 2. 熊猫都处于一个看起来像是模拟自然环境的场所，可能是动物园或保护区。
# # 3. 熊猫周围都有竹子，这是它们的主要食物来源。
# # 
# # 这两张图片的区别：
# # 
# # 1. 熊猫的种类不同：第一张图片是一只红熊猫，第二张图片是一只大熊猫。
# # 2. 熊猫的皮毛颜色和图案不同：红熊猫的皮毛是橙红色，脸部、耳朵和四肢的末端有白色斑块；而大熊猫的皮毛是黑白相间的，眼睛、耳朵和四肢的末端是黑色的，脸部、耳朵内侧和身体其他部分是白色的。
# # 3. 熊猫的姿态和位置不同：红熊猫坐在一个木制的结构上，而大熊猫坐在地上。
# # 4. 背景中的植被和环境细节略有不同，但都包含树木和竹子。
