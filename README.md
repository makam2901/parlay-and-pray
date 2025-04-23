# parlay-and-pray
AI-powered betting assistant using ML models for predictive insights. Features an LLM front-end for natural language queries, making analytics accessible. Focuses on a single sport first, expanding as feasible. Enhances decision-making by bridging data and action for bettors. Aims to provide data-driven recommendations with strong market potential.

# Setup (sample project)

## Folder Setup

- clone the repo
```bash
git clone https://github.com/makam2901/parlay-and-pray.git
```
- Copy all the contents to a local folder `parlay-and-pray-local`

- create a folder `data`

- Add the data file in it `heart-disease.csv`. Access it [here](https://drive.google.com/open?id=1-3FGXqQn2FOPDx9YNEAdVW3H2cvRzcLu&usp=drive_fs).

## ML Flow Setup

- Run the script on terminal
```bash
python mlflow_setup/mlflow_server.py
```

- Access it [here](http://localhost:5001)

## Test Training Flow
- Run the script on Terminal
```bash
python src/training_flow.py run
```

## Test Scoring Flow
```bash
python src/scoring_flow.py run --vector '[63,1,3,145,233,1,0,150,0,2.3,0,0,1]'
```
