# Base image with Python 3.8
FROM python:3.11.9

# Set the working directory
WORKDIR /app

# Copy your Streamlit code and requirements into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_ui.py"]
