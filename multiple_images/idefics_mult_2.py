import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

from jinja2 import Template

'''
IMAGES - 
"https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG" -> Idefics - Running Dog
"https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052" -> Asterix - Man with red cape
"https://llava-vl.github.io/static/images/view.jpg" -> View of a lake surrounded by forest
'''

prompt1 = [
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

prompt2 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is the target",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

prompt3 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is the target",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]


prompt4 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is the target",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The second image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

prompt5 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The second image is the target",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

prompt6 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The second image is the target",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: The third image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]



prompt7 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is the target",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is a distractor",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The third image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

prompt8 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is the target",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The third image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "User: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text",
        "<end_of_utterance>",
        "\nAssistant:",
    ],
]

prompt9 = [
    [
        "User: You are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images",
        "User: The first image is a distractor",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "User: The second image is the target",
        "https://llava-vl.github.io/static/images/view.jpg",
        "User: The third image is a distractor",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
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
print("##################### Expected output - Running Dog")
generate_output(prompt6)
print("##################### Expected output - Running Dog")
generate_output(prompt7)
print("##################### Expected output - Lake view")
generate_output(prompt8)
print("##################### Expected output - Man with red cape")
generate_output(prompt9)


'''
RESULTS:
'''


