from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

model = AutoModelForVision2Seq.from_pretrained('xtuner/llava-llama-3-8b-v1_1-hf')
processor = AutoProcessor.from_pretrained('xtuner/llava-llama-3-8b-v1_1-hf')

image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

print("GENERAL CASE ############################################################")
prompts = "USER: <image>\nWhat is shown in the image?\nASSISTANT:"
inputs = processor(prompts, images=[image1], return_tensors="pt").to("cuda")

output = model.generate(**inputs, max_new_tokens=200)
generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)
