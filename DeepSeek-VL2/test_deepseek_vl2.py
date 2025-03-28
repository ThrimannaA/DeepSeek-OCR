# from transformers import AutoProcessor, AutoModelForVision2Seq
# from PIL import Image
# import torch
# import cv2
# import numpy as np

# def preprocess_image(image_path):
#     """Preprocess image for better OCR accuracy."""
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
#     _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply thresholding
#     return Image.fromarray(thresh)  # Convert back to PIL format

# # Load the DeepSeek OCR model
# processor = AutoProcessor.from_pretrained("deepseek-ai/deepseek-vision-ocr")
# model = AutoModelForVision2Seq.from_pretrained("deepseek-ai/deepseek-vision-ocr")

# # Load and preprocess the image
# image_path = "0012.png"  # Change this to your actual image filename
# try:
#     image = preprocess_image(image_path)
# except Exception as e:
#     print(f"Error loading image: {e}")
#     exit()

# # Process image
# inputs = processor(images=image, return_tensors="pt")

# # Generate OCR output
# with torch.no_grad():
#     generated_ids = model.generate(**inputs)
#     extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# # Post-process extracted text
# extracted_text = extracted_text.replace("\n", " ").strip()  # Remove unnecessary line breaks

# print("Extracted OCR Text:", extracted_text)





# from deepseek_vl2.models import VLChatProcessor, DeepSeekVLForConditionalGeneration
# from PIL import Image
# import torch

# # Load model locally
# processor = VLChatProcessor.from_pretrained("./pretrained_models")
# model = DeepSeekVLForConditionalGeneration.from_pretrained("./pretrained_models")

# # Use CPU or GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # Test OCR on one image
# image_path = "../dataset/images/1331102.png"
# image = Image.open(image_path).convert("RGB")
# # inputs = processor(images=image, text="Extract all text from this image", return_tensors="pt").to(device)
# # inputs = processor(images=image, text="Extract whole texts from image with those spaces and bullets", return_tensors="pt").to(device)
# inputs = processor(images=image, text="Extract all text from this image exactly as it appears, preserving spaces, bullets, numbers, and all formatting", return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=512)  # Ensure full text
# text = processor.decode(outputs[0], skip_special_tokens=True)
# print("Extracted Text:", text)





# from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
# from PIL import Image
# import torch

# # Load model locally
# processor = DeepseekVLV2Processor.from_pretrained("./pretrained_models")
# model = DeepseekVLV2ForCausalLM.from_pretrained("./pretrained_models")

# # Use CPU or GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# # Test OCR on one image
# image_path = "../dataset/images/1331102.png"
# image = Image.open(image_path).convert("RGB")
# inputs = processor(
#     images=image,
#     text="Extract all text from this image exactly as it appears, preserving spaces, bullets, numbers, and all formatting",
#     return_tensors="pt"
# ).to(device)
# outputs = model.generate(**inputs, max_new_tokens=512)  # Ensure full text
# text = processor.decode(outputs[0], skip_special_tokens=True)
# print("Extracted Text:", text)



from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from PIL import Image
import torch

print("Starting script...")
# Load model locally
print("Loading processor...")
processor = DeepseekVLV2Processor.from_pretrained("./pretrained_models")
print("Loading model...")
model = DeepseekVLV2ForCausalLM.from_pretrained("./pretrained_models")
print("Model loaded, moving to device...")

# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model ready.")

# Test OCR on one image
image_path = "../dataset/images/1331102.png"
print(f"Opening image: {image_path}")
image = Image.open(image_path).convert("RGB")
print("Image loaded, processing...")
inputs = processor(
    images=image,
    text="Extract all text from this image exactly as it appears, preserving spaces, bullets, numbers, and all formatting",
    return_tensors="pt"
).to(device)
print("Inputs prepared, generating text...")
outputs = model.generate(**inputs, max_new_tokens=512)
print("Text generated, decoding...")
text = processor.decode(outputs[0], skip_special_tokens=True)
print("Extracted Text:", text)