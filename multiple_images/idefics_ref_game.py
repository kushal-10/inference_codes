import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image

image3 = Image.open("images/3.png")
image8 = Image.open("images/8.png")
image13 = Image.open("images/13.png")

prompt1 = [
    [
        'User: \nYou are given three images, one is called target and the other two are distractors.\nYour task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.\nThe first image is the target, \nthe second image is a distractor, and \nthe third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag "Expression: " for the given target image. Omit any other text',
        image3,
        image8,
        image13,
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

print("##################### Expected output - First row is empty other rows are filled with Xs")
generate_output(prompt1)

'''
RESULT1
User: You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor imagesUser: The first image is the targetimages/3.pngUser: The second image is a distractorimages/8.pngUser: The third image is a distractorimages/13.pngUser: Instruction: Describe the target image. Generate the referring expression starting with the tag 'Expression: ' for the given target image. Omit any other text 
Assistant: Expression: The target image shows a man wearing a red shirt and black pants, holding a tennis racket in his hand.

RESULT2
0:
User: What can you see in this image?images/3.png 
Assistant: I'm sorry, but the image is not available for me to see. Please provide more information or context about the image.

Maybe it only accepts an online path, and never a local path
It also accepts PIL.Image object, try that.

Passing as an object worked
0:
User: What can you see in this image? 
Assistant: The image is a black and white grid with 25 squares, arranged in a 5x5 pattern. Each square is either filled with a black or white color.

Try multimodal refernce game instance
X|
User: 
You are given three images, one is called target and the other two are distractors.
Your task is to generate a referring expression that best describes the target image while distinguishing it from the two other distractor images.
The first image is the target, 
the second image is a distractor, and 
the third image is a distractor. Instruction: Describe the target image. Generate the referring expression starting with the tag "Expression: " for the given target image. Omit any other text 
Assistant: Expression: The target image is a grid with 25 squares, 5 rows and 5 columns.

'''