# Project Setup Guide: Heart Disease Chatbot (Dockerized)

**Date:** April 22, 2025

## Overview

This project implements a chatbot application that can answer general questions and provide a preliminary prediction for heart disease based on user-provided health metrics. It uses:
* **Frontend:** Streamlit
* **Backend API:** FastAPI
* **LLM Interaction:** LangChain with Ollama (running the Mistral model)
* **ML Model:** Scikit-learn model trained on heart disease data
* **MLOps:** MLflow for experiment tracking and model registry, Metaflow for training orchestration
* **Containerization:** Docker and Docker Compose

This guide explains how to set up and run the complete application locally using Docker Compose.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Git:** For cloning the repository. ([Download Git](https://git-scm.com/downloads))
2.  **Docker Desktop:** Includes Docker Engine and Docker Compose. Ensure Docker Desktop is running. ([Download Docker Desktop](https://www.docker.com/products/docker-desktop/))
3.  **Ollama & Mistral Model:** The backend requires access to an Ollama instance running the `mistral` model.
    * **Install Ollama:** Follow the instructions at [https://ollama.com/](https://ollama.com/).
    * **Download Mistral Model:** Once Ollama is installed, run the following command in your terminal:
        ```bash
        ollama pull mistral
        ```
    * **Run Ollama:** Ensure the Ollama application/server is running locally. The default setup expects Ollama to be accessible at `http://localhost:11434` *from your host machine*.

    * **Note:** The `docker-compose.yml` configuration uses `http://host.docker.internal:11434` for the `OLLAMA_BASE_URL` environment variable in the `backend` service. This special DNS name allows the Docker container to reach the service running on your host machine (works on Docker Desktop for Mac/Windows, and often Linux with recent Docker versions). If your Ollama setup differs (e.g., running in a separate Docker container, different port, remote server), you will need to update the `OLLAMA_BASE_URL` value in `docker-compose.yml` accordingly.

## Setup Steps

1.  **Clone the Repository:**
    - run the command
        ```bash
        git clone https://github.com/makam2901/parlay-and-pray.git
        ```
    - Copy all the contents in `sample-chatbot` to a local folder `parlay-and-pray-local`

    - create a folder `data`

    - Add the data file in it `heart-disease.csv`. Access it [here](https://drive.google.com/open?id=1-3FGXqQn2FOPDx9YNEAdVW3H2cvRzcLu&usp=drive_fs).
    - On terminal go to the folder
        ```bash
        cd parlay-and-pray-local
        ```

2.  **Review Configuration (Optional):**
    The project includes pre-configured files:
    * `Dockerfile`: Defines the common container image.
    * `docker-compose.yml`: Orchestrates all the services.
    * `requirements.txt`: Lists Python dependencies.
    * `.dockerignore`: Specifies files/directories to exclude from the Docker build context.
    The main configuration you might need to adjust (as mentioned in Prerequisites) is the `OLLAMA_BASE_URL` in `docker-compose.yml` if your Ollama setup isn't running locally on the default port.

3.  **Build Docker Images:**
    This command builds the Docker image defined in the `Dockerfile`, installing all dependencies from `requirements.txt`. This image will be used by all services defined in `docker-compose.yml`.
    ```bash
    docker-compose build
    ```
    *(This might take a few minutes the first time)*

4.  **Run Initial Model Training:**
    This step runs the Metaflow training pipeline inside a temporary container. It uses the data in the `./data` directory, trains the model, and registers the best version with the MLflow server (which will be started automatically if not already running). **You only need to run this once initially** or if you update the training data or model code significantly.
    ```bash
    docker-compose run --rm training
    ```
    * `run`: Executes a one-off command in a service container.
    * `--rm`: Removes the container after the command finishes.
    * `training`: The name of the service defined in `docker-compose.yml` to run.
    This command implicitly starts the `mlflow` service first because `training` `depends_on` it. Wait for this command to complete successfully. You should see output indicating the model training process and final registration.

5.  **Run the Application Services:**
    This command starts all the long-running services defined in `docker-compose.yml` (MLflow server, Scoring API, Backend API, Frontend) in the background.
    ```bash
    docker-compose up -d
    ```
    * `-d`: Runs the containers in detached mode (in the background).

6.  **Verify Services are Running:**
    Check the status of the containers:
    ```bash
    docker-compose ps
    ```
    You should see entries for `mlflow_server`, `scoring_api`, `backend_api`, and `frontend_app`, all showing a state like `Up` or `Running`. The `mlflow` and `scoring` services should eventually show `(healthy)` in their status if the healthchecks are passing correctly.

## Accessing the Application

Once the services are running, you can access them via your web browser:

* **Frontend Chatbot:** [http://localhost:8501](http://localhost:8501) - This is the main user interface.
* **MLflow UI:** [http://localhost:5001](http://localhost:5001) - View training runs, metrics, parameters, and registered models.
* **Backend API Docs (Optional):** [http://localhost:8000/docs](http://localhost:8000/docs) - Interactive FastAPI documentation for the chat backend.
* **Scoring API Docs (Optional):** [http://localhost:8001/docs](http://localhost:8001/docs) - Interactive FastAPI documentation for the scoring service.

## Testing the Workflow

1.  Navigate to the **Frontend Chatbot** at `http://localhost:8501`.
2.  Engage in basic chat (e.g., "Hello", "What is angina?"). Verify responses from the LLM.
3.  Ask for a prediction using keywords like "predict", "check", "disease", "heart".
4.  The chatbot should prompt for 13 comma-separated values.
5.  Enter the example values: `63,1,3,145,233,1,0,150,0,2.3,0,0,1`
6.  Verify that you receive a prediction result message (e.g., "...preliminary check indicates: potential risk detected..."). This confirms the backend called the scoring API, which loaded the trained model from MLflow.

## Stopping the Application

When you are finished, stop and remove the containers and network created by Docker Compose:

```bash
docker-compose down
```
- Note on Data Persistence:
    -  The command `docker-compose down` does not delete the named volume mlflow_data. Your MLflow runs and registered models will still be there the next time you run docker-compose up -d.
    - If you want to remove the MLflow data as well (e.g., for a completely fresh start), use:
        ```bash
        docker-compose down -v
        ```
        (Use the -v flag with caution!)

## Troubleshooting


1. **Connection Refused Errors**: If one service can't connect to another (e.g., backend to scoring, or training/scoring to mlflow), ensure the target service is running and healthy (docker-compose ps). Check the logs of the failing service (docker-compose logs <service_name>) and the target service. Healthchecks in docker-compose.yml should prevent most startup-related connection issues.

2. **Service Unhealthy**: If docker-compose ps shows a service as unhealthy, check its logs (docker-compose logs <service_name>) to see why the healthcheck might be failing. Also, verify the healthcheck.test command in docker-compose.yml is correct for that service.

3. **Ollama Connection Issues**: If the backend logs show errors connecting to Ollama, double-check that Ollama is running on your host machine and accessible at the URL specified by OLLAMA_BASE_URL in the backend service definition in docker-compose.yml.

4. **"Read-only file system" Errors**: Usually indicates a volume was mounted with :ro but the container needs write access. Check the volumes definition in docker-compose.yml for the affected service.

5. **View Logs**: docker-compose logs <service_name> (e.g., docker-compose logs backend) is your best friend for debugging. Use docker-compose logs -f <service_name> to follow logs in real-time.

6. **Inspect Volumes**: To see where Docker is storing the persistent MLflow data on your host: docker volume inspect mlflow_data (You generally don't need to interact with this directory directly).