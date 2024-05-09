from PIL import Image
import requests


image3 = Image.open(requests.get("https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG", stream=True).raw)
image3 = Image.open(requests.get("https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052", stream=True).raw)
image3 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)

# Resize images to have the same height
min_height = min(image1.height, image2.height, image3.height)
image1 = image1.resize((int(image1.width * min_height / image1.height), min_height))
image2 = image2.resize((int(image2.width * min_height / image2.height), min_height))
image3 = image3.resize((int(image3.width * min_height / image3.height), min_height))

combined_width = image1.width + image2.width + image3.width

# Create a new blank image 
combined_image = Image.new("RGB", (combined_width, min_height))

# Paste each image
combined_image.paste(image1, (0, 0))
combined_image.paste(image2, (image1.width, 0))
combined_image.paste(image3, (image1.width + image2.width, 0))

combined_image.save("combined_image.jpg")