from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from io import BytesIO
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from fastapi.responses import JSONResponse

app = FastAPI()

# ตรวจสอบว่า GPU พร้อมใช้งาน
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# โหลดโมเดลบน GPU ด้วย dtype ที่เหมาะสม
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "scb10x/typhoon-ocr-7b", 
    torch_dtype=dtype,
    device_map="auto"  # หรือจะใช้ .to(device) ด้านล่างก็ได้
).eval()

# โหลด processor
processor = AutoProcessor.from_pretrained("scb10x/typhoon-ocr-7b")

@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    text: str = Form("Extract all text")  # default prompt
):
    image = Image.open(BytesIO(await file.read())).convert("RGB")

    # สร้าง prompt
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    # เตรียม input และย้ายไป GPU
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device=device, dtype=dtype)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        # decode เฉพาะส่วนที่ถูก generate ใหม่
        result = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

    return JSONResponse(content={
        "success": True,
        "filename": file.filename,
        "extracted_text": result.strip(),
        "prompt_used": text
    })
