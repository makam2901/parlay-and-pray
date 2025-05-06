# Guide to GCP Deployment - Cross functional
## Github
- repo: [parlay-and-pray](https://github.com/makam2901/parlay-and-pray/)
- Copy all the contents of this folder into your local folder.
- Navigate to this folder as it is your project-folder for instructions below.

## Permissions
These are instructions for the owner only (manikesh) to grant access to teammates

1. configure
    ```bash
    gcloud components install kubectl
    gcloud auth login
    gcloud config set project parlay-and-pray-app
    ```

2. project access
    ```bash
    PROJECT_ID="parlay-and-pray-app"
    TEAMMATE_EMAIL="manikeshmakam@gmail.com"

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${TEAMMATE_EMAIL}" \
    --role="roles/editor"
    ```

3. development access
    ```bash
    PROJECT_ID="parlay-and-pray-app"
    TEAMMATE_EMAIL="manikeshmakam@gmail.com"

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${TEAMMATE_EMAIL}" \
    --role="roles/artifactregistry.writer"
    ```

4. deployment access
    ```bash
    PROJECT_ID="parlay-and-pray-app"
    TEAMMATE_EMAIL="manikeshmakam@gmail.com"

    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="user:${TEAMMATE_EMAIL}" \
    --role="roles/container.developer"
    ```

## Deployment 
These are the instructions for any to set the GCP hosted external IP address link.

1. configure
    ```bash
    gcloud auth login
    gcloud config set project parlay-and-pray-app
    gcloud config set compute/region us-central1
    gcloud container clusters get-credentials autopilot-cluster-1 --region us-central1
    ```

2. apply manifests
    ```bash
    kubectl apply -f backend-deployment.yaml
    kubectl apply -f backend-service.yaml
    kubectl apply -f frontend-deployment.yaml
    kubectl apply -f frontend-service.yaml
    ```

3. check status
    ```bash
    kubectl get pods --namespace default -w
    ```

4. verification
    - Wait until `STATUS` is `Running` for both frontend and backend pods. Press Ctrl+C to exit watch. 
    - If issues arise (e.g., ImagePullBackOff, CrashLoopBackOff), use kubectl describe pod <pod-name> and kubectl logs <pod-name> to investigate.
    - `control + C` to exit

5. live link
    - get External IP for frontend
    ```bash
    kubectl get service frontend-service --namespace default --watch
    ```
    - Wait for an IP address to appear in the `EXTERNAL-IP` column.
    - `control + C` to exit
    - Wait for a while for initial model to run on stratup. (check logs in backend deployment on console)
    - Refresh if there is any issue. If issue still exists reach out to the owner.

6. termination
    - at the end of the day, delete the deployment and servbices to save costs
    ```bash
    kubectl delete service frontend-service --namespace default
    kubectl delete service backend-service --namespace default
    kubectl delete deployment frontend-deployment --namespace default
    kubectl delete deployment backend-deployment --namespace default
    ```

## Development
These are the instructions for anyone to push an updated image to artifact repo if there is any change in code or the application.

Do everything from project folder.

1. configure
    - run your docker desktop
    - run the following commands
        ```bash
        gcloud auth login
        gcloud config set project parlay-and-pray-app
        gcloud config set compute/region us-central1
        gcloud container clusters get-credentials autopilot-cluster-1 --region us-central1
        gcloud auth configure-docker us-central1-docker.pkg.dev
        ```
    - install buildx
        ```bash
        docker buildx create --name mybuilder --use
        docker buildx inspect --bootstrap
        ```

2. code
    - go to project_folder (look for project folder structure below. It should follow the exact names and structure)
    - make changes to the code

3. build and push the image
    - backend image
        ```bash
        cd src/backend
        docker buildx build --platform linux/amd64,linux/arm64 \
        -t us-central1-docker.pkg.dev/parlay-and-pray-app/mlops-dream11-repo/backend-app:v4 \
        --push .
        cd ../..
        ```
    - frontend image
        ```bash
        cd src/frontend
        docker buildx build --platform linux/amd64,linux/arm64 \
        -t us-central1-docker.pkg.dev/parlay-and-pray-app/mlops-dream11-repo/frontend-app:v4 \
        --push .
        cd ../..
        ```
4. deployment
    - follow the steps from above.