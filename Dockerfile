FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir \
    --timeout=1000 \
    --retries=10 \
    -r requirements-docker.txt

COPY src/ ./src/
COPY api/ ./api/
COPY inference/ ./inference/
COPY .env .

EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]