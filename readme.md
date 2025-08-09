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
