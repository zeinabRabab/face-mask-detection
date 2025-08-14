# 1. Use a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire app into the container
COPY . .

# 6. Expose port 5000 for Flask
EXPOSE 5000

# 7. Run Flask app
CMD ["python", "app.py"]


