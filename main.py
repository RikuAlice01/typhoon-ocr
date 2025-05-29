from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from io import BytesIO
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
from typing import List, Dict, Any

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Typhoon OCR API (Optimized)",
    description="High-performance OCR API using Typhoon OCR 7B model with optimizations",
    version="1.0.0"
)

# Global variables for model and processor
model = None
processor = None
device = None
thread_pool = None

def optimize_gpu_settings():
    """ปรับแต่งการใช้งาน GPU เพื่อประสิทธิภาพสูงสุด"""
    if torch.cuda.is_available():
        # เปิด TensorFloat-32 (TF32) สำหรับ Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # เปิด cudnn benchmark สำหรับ input sizes ที่คงที่
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # ปรับแต่ง memory management
        torch.cuda.empty_cache()
        
        logger.info("GPU optimizations applied")

def precompile_model():
    """Pre-compile model สำหรับ inference ที่เร็วขึ้น"""
    global model, device
    if model is not None and torch.cuda.is_available():
        try:
            # Warm up model with dummy input
            dummy_input = torch.randint(0, 1000, (1, 100)).to(device)
            with torch.no_grad():
                _ = model(input_ids=dummy_input)
            logger.info("Model warm-up completed")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

@app.on_event("startup")
async def startup_event():
    """โหลดโมเดลเมื่อเริ่มต้น server พร้อมการปรับแต่ง"""
    global model, processor, device, thread_pool
    
    try:
        logger.info("กำลังโหลดโมเดลพร้อมการปรับแต่ง...")
        
        # ตรวจสอบ system resources
        logger.info(f"Available CPU cores: {psutil.cpu_count()}")
        logger.info(f"Available RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # ตั้งค่า thread pool สำหรับ async operations
        thread_pool = ThreadPoolExecutor(max_workers=min(4, psutil.cpu_count()))
        
        # ตรวจสอบ CUDA และปรับแต่ง
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
            optimize_gpu_settings()
        logger.info(f"Using device: {device}")
        
        # โหลดโมเดลและ processor พร้อมการปรับแต่ง
        start_time = time.time()
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "scb10x/typhoon-ocr-7b", 
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        
        processor = AutoProcessor.from_pretrained(
            "scb10x/typhoon-ocr-7b", 
            use_fast=True,
            padding_side="left"
        )
        
        # Pre-compile และ warm up model
        precompile_model()
        
        load_time = time.time() - start_time
        logger.info(f"โหลดโมเดลสำเร็จ! ใช้เวลา {load_time:.2f} วินาที")
        
        # แสดงข้อมูล memory usage
        if torch.cuda.is_available():
            logger.info(f"GPU Memory used: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            logger.info(f"GPU Memory cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """ทำความสะอาดเมื่อปิด server"""
    global model, processor, thread_pool
    
    if thread_pool:
        thread_pool.shutdown(wait=True)
    
    if model is not None:
        del model
    if processor is not None:
        del processor
    
    # ทำความสะอาด memory อย่างละเอียด
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.info("ทำความสะอาด resources สำเร็จ")

def optimize_image(image: Image.Image) -> Image.Image:
    """ปรับแต่งรูปภาพเพื่อประสิทธิภาพที่ดีขึ้น"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    max_size = 620
    ratio = min(max_size / image.width, max_size / image.height)
    if ratio < 1:
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

async def perform_ocr_optimized(image: Image.Image, prompt: str) -> str:
    """ทำ OCR ที่ปรับแต่งแล้วเพื่อความเร็ว"""
    try:
        start_time = time.time()
        
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(thread_pool, optimize_image, image)
        
        conversation = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        inputs = {key: value.to(device, non_blocking=True) for key, value in inputs.items()}

        with torch.no_grad():
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    compiled_generate = torch.compile(model.generate, mode="reduce-overhead")
                    output = compiled_generate(
                        **inputs,
                        max_new_tokens=800,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                        num_beams=1,
                    )
                except Exception:
                    output = model.generate(
                        **inputs,
                        max_new_tokens=800,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        use_cache=True,
                        early_stopping=True,
                        num_beams=1,
                    )
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=800,
                    temperature=0.1,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    early_stopping=True,
                    num_beams=1,
                )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_len:]
        decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        processing_time = time.time() - start_time
        logger.info(f"OCR processing completed in {processing_time:.2f} seconds")

        return decoded[0].strip()

    except Exception as e:
        logger.error(f"Error in OCR processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr/general")
async def ocr_general(
    file: UploadFile = File(...),
    prompt: str = "Extract all text from this image"
):
    """
    ทำ OCR ทั่วไปกับรูปภาพ
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        result = await perform_ocr_optimized(image, messages)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "extracted_text": result,
            "prompt_used": prompt
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.post("/ocr/custom")
async def ocr_custom(
    file: UploadFile = File(...),
    messages: List[Dict[str, Any]] = None
):
    """
    ทำ OCR ด้วย custom messages
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if messages is None:
        messages = [
            {
                "role": "user",
                "content": "Extract all text from this image"
            }
        ]
    
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        result = await perform_ocr_optimized(image, messages)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "extracted_text": result,
            "messages_used": messages
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing with custom messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")

@app.get("/")
async def root():
    """หน้าแรกของ API"""
    return {
        "message": "Typhoon OCR API (Optimized)",
        "version": "1.0.0",
        "endpoints": {
            "/ocr/general": "POST: Upload image and optional prompt string",
            "/ocr/custom": "POST: Upload image and custom messages in JSON",
            "/healthz": "GET: Health check"
        }
    }

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}
