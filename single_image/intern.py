import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
image = ['./examples/cars1.jpg',
        './examples/cars2.jpg',
        './examples/cars3.jpg',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
#To analyze the advantages and disadvantages of each car, we need to consider factors such as brand reputation, performance, design, cost, and maintenance.
#1. Mercedes-Benz: - Advantages: Known for its luxury and reliability, the Mercedes-Benz brand offers a high level of comfort, advanced technology, and superior craftsmanship. The vehicle in the image appears to be an SUV, which is versatile for both city driving and off-road conditions. - Disadvantages: Typically, Mercedes-Benz vehicles are more expensive compared to other brands, and they may require more frequent maintenance due to their luxury status.
#2. Bugatti: - Advantages: Bugatti is renowned for producing some of the fastest and most powerful cars in the world. The vehicle in the image is a sports car, likely offering an exhilarating driving experience with its high-performance engine and advanced aerodynamics. - Disadvantages: Bugatti cars are extremely expensive, making them less accessible to the average consumer. They also require specialized knowledge for maintenance and may not be suitable for everyday driving due to their high-performance nature.
#3. BMW: - Advantages: BMW is a well-respected manufacturer known for its innovative engineering and design. The car in the image appears to be an electric vehicle (EV), which offers advantages such as lower emissions, reduced noise pollution, and potentially lower operating costs. - Disadvantages: EVs like the one in the image may have a shorter range compared to traditional gasoline-powered vehicles, limiting their suitability for long-distance travel.
#Additionally, the initial cost of purchasing an EV can be higher than that of a conventional car. In conclusion, each car has its own set of advantages and disadvantages. The Mercedes-Benz offers luxury and versatility, the Bugatti provides high performance and exclusivity, and the BMW represents innovation in electric vehicle technology. The choice ultimately depends on the buyer's priorities, such as comfort, performance, environmental impact, and budget.

query = 'Image4 <ImageHere>; How about the car in Image4'
image.append('./examples/cars4.jpg')
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, history= his, use_meta=True)
print(response)
#The car in Image4 is a red sports car, which appears to be a Ferrari. Ferrari is a renowned Italian manufacturer known for producing some of the most iconic and high-performance sports cars in the world. - Advantages: Ferrari vehicles are synonymous with speed, luxury, and engineering excellence.
#The car in the image likely offers an exhilarating driving experience with its powerful engine, advanced aerodynamics, and high-quality craftsmanship. The red color adds to the car's aesthetic appeal, making it stand out on the road. - Disadvantages: Ferrari cars are extremely expensive, making them less accessible to the average consumer.
#They also require specialized knowledge for maintenance and may not be suitable for everyday driving due to their high-performance nature. In conclusion, the Ferrari in Image4 represents a pinnacle of automotive engineering and design, offering unmatched performance and luxury.
#However, its high cost and specialized maintenance requirements make it less practical for everyday use compared to the other vehicles in the images.
