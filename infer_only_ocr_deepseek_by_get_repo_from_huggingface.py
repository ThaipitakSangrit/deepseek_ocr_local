from transformers import AutoModel, AutoTokenizer
import torch
import os
from glob import glob
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "

image_folder = r'D:/work/python/ocr_utl3_test_infer_deepseek_ocr/images'
result_path = r'D:/work/python/ocr_utl3_test_infer_deepseek_ocr/results'

os.makedirs(image_folder, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

# อ่านไฟล์รูปทั้งหมด (รองรับ .jpg, .png, .jpeg)
image_files = []
for ext in ('*.jpg', '*.jpeg', '*.png'):
    image_files.extend(glob(os.path.join(image_folder, ext)))

for i, image_file in enumerate(image_files, 1):
    print(f"[{i}/{len(image_files)}] กำลังประมวลผล: {os.path.basename(image_file)} ✅")
    
    image = Image.open(image_file).convert('RGB')
    w, h = image.size

    # เช็คขนาดรูป
    if (w, h) == (640, 480):
        print(f"   🔍 พบขนาด 640x480 → ขยายเป็น {w*2}x{h*2}")
        image = image.resize((w * 2, h * 2), Image.NEAREST)
        image.save(image_file)
    else:
        print(f"   ✅ ขนาดปกติ ({w}x{h})")

    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = result_path + '/' + os.path.basename(image_file), base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)