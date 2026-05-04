FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt pyproject.toml ./
COPY noshow_iq/ ./noshow_iq/
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir .

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY noshow_iq/ ./noshow_iq/
COPY train_model.py ./train_model.py
COPY data/ ./data/
ENV MODEL_PATH=/app/noshow_model.joblib
RUN chown -R appuser:appuser /app
USER appuser
RUN python train_model.py
EXPOSE 7860
CMD ["python", "-m", "noshow_iq.api"]