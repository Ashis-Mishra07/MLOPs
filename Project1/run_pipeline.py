from pipelines.training_pipeline import train_pipeline

# to the all the prev model through the pipelines 
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri()) # this will show u the URLs
    train_pipeline(data_path="F:\Complete ML\MLOPs\Project1\data\olist_customers_dataset.csv")


    
    