import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

from jinja2 import Template

'''
IMAGES - 
"https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG" -> Idefics - Running Dog
"https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052" -> Asterix - Man with red cape
"https://llava-vl.github.io/static/images/view.jpg" -> View of a lake surrounded by forest
'''
# Single User message - Default
prompt1 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.\n<image>\nThe first image is a distractor, <image>\nthe second image is the target, and <image>\nthe third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "https://llava-vl.github.io/static/images/view.jpg",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

# Single User message - Default - Change order
prompt2 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.\n<image>\nThe first image is the target, <image>\nthe second image is a distractor, and <image>\nthe third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "https://llava-vl.github.io/static/images/view.jpg",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

# Single User message - Default - Change order again
prompt3 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.\n<image>\nThe first image is a distractor, <image>\nthe second image is a distractor, and <image>\nthe third image is the target. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "https://llava-vl.github.io/static/images/view.jpg",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]


# Different user message, for each image
prompt4 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is the target",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

# Change order
prompt5 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is the target",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-80b-instruct"
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


def generate_output(prompt):
    # --batched mode
    inputs = processor(prompt, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    # --single sample mode
    # inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=512)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")

print("##################### Expected output - Man with red cape")
generate_output(prompt1)
print("##################### Expected output - Cartoon Dog")
generate_output(prompt2)
print("##################### Expected output - Lake view")
generate_output(prompt3)
print("##################### Expected output - Lake view")
generate_output(prompt4)
print("##################### Expected output - Man with red cape")
generate_output(prompt5)

'''
RESULTS:
##################### Expected output - Man with red cape
0:
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.
 
The first image is a distractor,  
the second image is the target, and  
the third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: The target image shows a dog running on a track.

##################### Expected output - Cartoon Dog
0:
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.
 
The first image is the target,  
the second image is a distractor, and  
the third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: The target image shows a cartoon dog running on a track.

##################### Expected output - Lake view
0:
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.
 
The first image is a distractor,  
the second image is a distractor, and  
the third image is the target. Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: The target image shows a dog running on a track.

##################### Expected output - Lake view
0:
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor imagesUser: The first image is a distractor User: The second image is a distractor User: The third image is the target User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: A wooden dock is on a lake with mountains in the background.

##################### Expected output - Man with red cape
0:
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor imagesUser: The first image is the target User: The second image is a distractor User: The third image is a distractor User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: The dog is running on the ground.

'''


