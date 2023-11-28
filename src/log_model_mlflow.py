# %%
import numpy as np
import mlflow
from tensorflow import keras

from mlflow.types.schema import Schema, TensorSpec
from mlflow.models import ModelSignature
from main import run_deep_learning


# %%

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Demo_Experiment")
mlflow.set_tag("Info", "Simple CNN for MNIST hand-written digit classification.")


# %%
RUN_NAME = "MNIST_small_CNN"
ARTIFACT_PATH = "demo_artifacts"
REGISTERED_MODEL_NAME = "the_first_model"

# the first dimension must be a variable dimension
input_schema = Schema(
    [
        TensorSpec(np.dtype(np.float32), (-1, 28, 28, 1), "x_train"),
        TensorSpec(np.dtype(np.float32), (-1, 10), "y_train"),
    ]
)
signature = ModelSignature(inputs=input_schema)

# Initiate the MLflow run context
with mlflow.start_run(run_name=RUN_NAME) as run:

    loss, acc, model = run_deep_learning()

    mlflow.log_metric("loss", loss)
    mlflow.log_metric("acc", acc)

    # Log an instance of the trained model for later use
    mlflow.tensorflow.log_model(
        model=model,
        signature=signature,
        registered_model_name=REGISTERED_MODEL_NAME, 
        artifact_path=ARTIFACT_PATH
        )


# %%
mlflow.end_run()
# %%
