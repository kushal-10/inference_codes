# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
print("LLM Success")


from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
model - AutoModelForVision2Seq.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")