FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --system --no-cache -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "server:app"]