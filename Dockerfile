FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY text_classifier ./text_classifier
COPY web_app ./web_app
COPY examples ./examples

ENV TRANSFORMER_STORAGE_DIR=/data
ENV TRANSFORMER_REQUIRE_API_KEY=1
EXPOSE 8765
CMD ["sh", "-c", "python -m uvicorn web_app.server:app --host 0.0.0.0 --port ${PORT:-8765}"]
