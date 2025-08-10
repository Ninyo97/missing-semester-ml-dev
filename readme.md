### Step 0: Problem Statement

### Step 0.5: What is $f_\theta$

### Step 1: torch.utils.data.Dataset

Dataset: A collection of samples $\{(x_i, y_i)\}$.

### Step 2: torch.utils.data.Dataloader

The Dataloader handles batching and dataset shuffling. A dataset is wrapped with a dataloader object.

### Step 3: Pytorch setup

- Tensors and `dir(torch.Tensor)`.
- The model is always implemented as a class that inherits `torch.nn.Module` object. This way, the model inherits a lot of internal methods and objects useful for ML (weights, backward and forward calls!).
- We generally need to implement the __init__ and forward functions to implement a model.
- Explore the model. Checkout `dir(model)`.

### Step 4: Train Pipeline

The train pipeline in it's most barebones form is a for loop over epochs.

- `Epoch`: One round of the model _training_ on _each_ sample in the train dataset. 
- `Batch`: A small subset of the training data, fed to the model in parallel is a batch.

_Honorable mentions_: `criterion`, `optimizer` and `loss.backwards`.

### Step 5: Evaluation Pipeline

- The model is evaluated on the test set at the end of training run.
- The performance of the model is evaluated based on a metric, typically different from the loss.

### Step 6: Refactor code

Time to move code around and give our project directory some structure.

Let us move the model to a `/models` folder. Also, we will move the dataset loading code to `datasets.py`. The reasons for these changes are for maintaining the code better. When we want to add a new model, we can just add a new file in the `/models` folder. Similarly, if we want to add a new dataset, we can just add a new function in `datasets.py`.


TIPS:
1. Use relative directory address.
2. Keep as little entry points as possible and keep all entry points in the same folder.
3. Let me stress point 2 by showing that the addresses `./MNIST/train` and `./MNIST/test` will work even if I mode the `datasets.py` to a subfolder `dataset`. This is because the entry point is from the root of the project directory.
4. `__init__.py` is a file that tells Python that the folder is a package. It can be empty, but it is required for Python to recognize the folder as a package. If you create a `__init__.py` file, then you can even write `import models`, for example. This is not necessary.


### Step 7: Configs

- Configs parsing is a generally large refactor. Function definitions change, variables get introduced. One needs to be careful about config parsing but it hugely increases the usability of the code.
- Currently, we have added a few training arguments for config parsing.
- There are many different ways of handling configs, using `hydra`, `OmegaConf` and more libraries.


### Step 8: Reproduciblity

- If you notice, all the runs have different loss values.
- There are a bunch of reasons for this such as model initialization, random batch shuffling, randomness involved in loss computation algorithms and so on.
- Reproducibility is generally achieved through seeding all the random number generators.


### Step 9: Logging

- Let us now introduce logging.
- Logging is more than just printing messages on the console. A log of an experiment specifies all the details of the experiment such that an experiment can be exactly tracked uing the log.

TIPS:
1. Log files are generally timestamped.
2. It is a good idea to mention all the configs at the start of the log file.
3. Some people print the model architecture in the log.
4. Log epochwise parameters in the log.


### Step 10: Model Checkpointing

- Storing the final model is important for future use.
- This process is also known as checkpointing. 
- Storing a model can be arbitrarily complex. One can store the model weights, the optimizer state, num_epochs and other metadata.
- We will simply store the model weights and the configurations. These weights can be used in future to load the model and use for deployment or finetune.


### Step 11: Early Stopping

- The last model, i.e. the model obtained after the final epoch, is often not the best model.
- It can be overfit, and may have a worse performance on the test set even though it has lower training error.
- To implement early stopping, we must have a validation set that we can test the model performance against.
- Moreover, the training might converge well before the `max_epochs` value. We can implement early stopping in this case. 

