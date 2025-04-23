from metaflow import FlowSpec, step
from train_model import train

class TrainFlow(FlowSpec):

    @step
    def start(self):
        print("Starting training flow...")
        self.next(self.train_model)

    @step
    def train_model(self):
        train()
        print("Model trained and logged with MLflow.")
        self.next(self.end)

    @step
    def end(self):
        print("Training flow completed.")

if __name__ == "__main__":
    TrainFlow()