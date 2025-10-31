from transformers import AutoTokenizer, AutoModel

# โฟลเดอร์ปลายทางที่ต้องการเซฟโมเดล (เปลี่ยน path ได้)
save_dir = "./models/DeepSeek_OCR"

# โหลด tokenizer และบันทึกลงโฟลเดอร์
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

# โหลด model และบันทึกลงโฟลเดอร์
model = AutoModel.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
model.save_pretrained(save_dir)

print(f"Model and tokenizer saved to: {save_dir}")
