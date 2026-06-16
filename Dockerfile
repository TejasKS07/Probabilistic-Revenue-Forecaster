FROM python:3.10-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
