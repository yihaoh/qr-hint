FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y default-jre gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV CFLAGS="-Wno-error=incompatible-pointer-types"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
EXPOSE 8080

CMD ["python", "app.py"]
