FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/chroma_db /app/uploaded_papers /app/data

EXPOSE 8778

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8778/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "chat_app.py", "--server.address=0.0.0.0", "--server.port=8778"]