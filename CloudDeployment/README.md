# Image Classification with FastAPI and Streamlit

This project provides an API for image classification using the best-performing model loaded into a FastAPI framework. The API is containerized using Docker, and a Streamlit-based UI is used to interact with the API for predictions.

---

## Folder Structure

```
CloudDeployment/
├── streamlit/            # Streamlit UI application files
├── Dockerfile            # Dockerfile to containerize the API
├── README.md             # Project documentation
├── cnn_fastapi.py        # FastAPI implementation of the classification API
├── model.pth             # Trained model file
├── requirements.txt      # Dependencies for the API
```

---

## Features

- **FastAPI-based API** for image classification.
- **Streamlit UI** to upload images and view predictions.
- **Dockerized Deployment** for seamless setup and scaling.
- **Supports Probabilistic Outputs** showing predictions along with confidence levels.

---

## How to Use

### 1. Running the API

#### Step 1: Transfer Project Files to EC2
Copy all project files to your EC2 instance.

#### Step 2: Build and Run the Docker Container

Run the following commands on the EC2 instance:

```bash
# Navigate to the project folder
cd CloudDeployment

# Build the Docker image
sudo docker build -t image-classification-api .

# Run the container
sudo docker run -d -p 8000:8000 image-classification-api
```
The API will now be running on port 8000.

### 2. Running the Streamlit UI

Run the UI locally on your system or any other server:

```bash
# Navigate to the streamlit directory
cd CloudDeployment/streamlit

# Start the Streamlit app
streamlit run app.py
```

Make sure to update the API URL in the Streamlit app (`app.py`) to match the EC2 instance's public IP and port (e.g., `http://<ec2-public-ip>:8000`).

### 3. Security Group Configuration for EC2
Ensure that the EC2 instance's security group allows incoming HTTP connections (port 80 or your configured Streamlit port) from the internet.

---

## Docker Image

The API is containerized into an OCI-compliant Docker image and can be pulled from Docker Hub:

Docker Hub: [aurovind/deep-learning-project-image-classification](https://hub.docker.com/r/aurovind/deep-learning-project-image-classification)

To run the prebuilt Docker image:

```bash
sudo docker pull aurovind/deep-learning-project-image-classification
sudo docker run -d -p 8000:8000 aurovind/deep-learning-project-image-classification
```

---

## Example Workflow

1. Open the Streamlit UI in a browser.
2. Upload an image.
3. View the predicted label and associated probability.

---

## Dependencies

Install required Python libraries before running the project:

```bash
pip install -r requirements.txt
```

---

## Notes

- Update the API URL in the Streamlit UI file (`app.py`) to match the actual deployment URL and port.
- The API runs on port `8000` by default. This can be adjusted as needed in the Docker run command.
- Ensure your EC2 instance's security group allows access to the relevant ports for both the API and UI.

---

## Contribution

Feel free to fork this repository and contribute by submitting pull requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

