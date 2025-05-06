# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# Assuming you have a requirements.txt in src/frontend/
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 8501 available (default Streamlit port)
EXPOSE 8501

# Define environment variable for the backend API URL
# This will be overridden by the Kubernetes deployment
ENV API_URL="http://backend-service.default.svc.cluster.local:8000/predict_team/"

# Run streamlit_app.py when the container launches
# Use --server.port to match EXPOSE and --server.address to listen on all interfaces
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]