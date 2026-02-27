# Use a lightweight Python base image
FROM python:3.10-slim

WORKDIR /app

# Install dependencies (make sure Flask, pymongo, requests, tiktoken, python-dotenv, flask-cors are in here)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your engine code
COPY main.py prompts.py .

# Force the container to expose Hugging Face's required port
ENV PORT=7860
EXPOSE 7860

# Launch the engine
CMD ["python", "main.py"]