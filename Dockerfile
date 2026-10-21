# Document Forensics — Hugging Face Spaces (Docker SDK) / genel Docker dağıtımı
#
#   docker build -t document-forensics .
#   docker run -p 7860:7860 -e FORENSICS_WEIGHTS_REPO=<hf-kullanici>/<model-deposu> document-forensics
#
# * CPU-only PyTorch kurulur (CUDA tekerlekleri ~2 GB'tır ve ücretsiz CPU ortamında gereksizdir).
# * Ağırlıklar imaja kopyalanmamışsa (checkpoints/*.pth git'e girmez) ilk açılışta
#   FORENSICS_WEIGHTS_REPO veya FORENSICS_WEIGHTS_URL ile indirilir (src/runtime.py).
# * HF Spaces konteyneri UID 1000 ile çalıştırır; ev dizini yazılabilir olmalıdır
#   (EasyOCR modelleri ~/.EasyOCR, HF önbelleği ~/.cache altına iner).
FROM python:3.11-slim

# OpenCV'nin çalışma zamanı kütüphaneleri
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
