from deepseek_vl2.models import VLChatProcessor, DeepSeekVLForConditionalGeneration
from PIL import Image
import torch
import os

processor = VLChatProcessor.from_pretrained("./pretrained_models")
model = DeepSeekVLForConditionalGeneration.from_pretrained("./pretrained_models")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

input_dir = "../dataset/images/"
output_dir = "../output/"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    if img_file.endswith((".png", ".jpg")):
        image_path = os.path.join(input_dir, img_file)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text="Extract all text from this image", return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=512)
        text = processor.decode(outputs[0], skip_special_tokens=True)
        output_file = os.path.join(output_dir, img_file.rsplit(".", 1)[0] + ".txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Processed {img_file}")