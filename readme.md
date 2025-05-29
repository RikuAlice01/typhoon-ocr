# ⚡ Typhoon OCR API (Optimized)

High-performance Optical Character Recognition (OCR) API built on top of the [SCB10X Typhoon OCR 7B model](https://huggingface.co/scb10x/typhoon-ocr-7b), powered by FastAPI and Transformers.

Designed for scalable deployment and optimized for GPU acceleration and low-latency inference.

---

## 🚀 Features

- 🔍 General and custom prompt-based OCR
- 🧠 Powered by `Typhoon OCR 7B` (Multimodal vision-language model)
- ⚡ GPU optimization: TF32, cuDNN tuning, memory handling
- 📷 Efficient image resizing and format conversion
- 📤 Upload image via REST API with prompt or message support
- 📦 Fully containerizable for deployment

---

## 📦 Installation

### 1. Clone Repository

```bash
git clone https://github.com/your-org/typhoon-ocr-api.git
cd typhoon-ocr-api
````

### 2. Install Dependencies

Make sure Python 3.10+ is installed.

```bash
pip install -r requirements.txt
```

✅ For GPU support, ensure that you have CUDA and the proper `torch` version installed. Example:


> ```bash
> pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
> ```

---

## 🧪 Usage

### Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then open your browser at:
📍 [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📤 API Endpoints

### `POST /ocr/general`

OCR with a general prompt (default or custom string)

#### Request:

* `file`: image file (jpg, png, etc.)
* `prompt`: *optional*, custom instruction (default: "Extract all text from this image")

#### Example:

```bash
curl -X POST http://localhost:8000/ocr/general \
  -F "file=@./sample.jpg" \
  -F "prompt=Extract the ID number and name from the ID card"
```

---

### `POST /ocr/custom`

OCR with a structured prompt using role-based messages.

#### Request:

* `file`: image file
* `messages`: JSON list of chat-style messages

#### Example Body (in Swagger or Postman):

```json
[
  {
    "role": "user",
    "content": "Extract text in table format from the document"
  }
]
```

---

### `GET /healthz`

Simple health check endpoint. Returns `"ok"` if the model and processor are loaded correctly.

---

## ⚙️ Environment Requirements

* Python >= 3.10
* CUDA-enabled GPU (optional but recommended)
* Pytorch >= 2.1.0
* Transformers >= 4.40.0

---

## 📊 Logging

Application logs are printed to stdout using Python `logging`.
Includes:

* Model load time
* GPU availability and memory
* OCR process time

---

## 📌 Notes

* This API loads the `Typhoon OCR 7B` model and performs warm-up during startup.
* GPU optimization includes TF32 acceleration, cuDNN tuning, and `torch.compile` for inference.
* You can extend the API to support multiple models, batch processing, or file exports easily.

---

## 🧑‍💻 Author

Sitthichai S. (2025)

---

## 📄 License

MIT License © 2025

---