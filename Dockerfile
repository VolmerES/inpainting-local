FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# La RTX 5060 (Blackwell, sm_120) necesita PyTorch compilado con CUDA 12.8+
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
