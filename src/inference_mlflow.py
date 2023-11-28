# %%
from PIL import Image
from numpy import asarray
import mlflow
# %%
model_name = "the_first_model"
model_version = 1

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")

# %%  load the image

PATH_SAMPLE = "./img_33.jpg"
image = Image.open(PATH_SAMPLE)
img_sample = asarray(image)
img_sample = img_sample.reshape(1,28,28)

# %%
model.predict(img_sample)
# %%

