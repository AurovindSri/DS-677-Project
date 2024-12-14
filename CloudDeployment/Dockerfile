# Base image with Python 3.8
FROM python:3.11.9

# Set the working directory
WORKDIR /app

# Copy your FastAPI code and requirements into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI app port
EXPOSE 8085

# Command to run the FastAPI app
CMD ["uvicorn", "cnn_fastapi:app", "--host", "0.0.0.0", "--port", "8085"]

