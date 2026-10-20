FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock
COPY research research
COPY tests tests
COPY data data
COPY pytest.ini .
CMD ["uvicorn", "research.web:app", "--host", "0.0.0.0", "--port", "8000"]
