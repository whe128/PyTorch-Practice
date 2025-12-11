import torchvision.transforms as transforms
from PIL import Image

# define the data preprocessing pipeline
transform = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
		transforms.Normalize(
				mean = [0.4885, 0.456, 0.406],
				std = [0.229, 0.224, 0.225]
		)
])

# load image
image = Image.open('image.jpg')

# apply preprocessing
image_tensor = transform(image)
print(image_tensor.shape)          # will be [3, 128, 128]  channel, H, W
