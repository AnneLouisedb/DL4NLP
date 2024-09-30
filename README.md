# Detecting Machine Generated Text vs. Human Generated Text

python3 -m venv DL4NLP source DL4NLP/bin/activate pip install -r requirements.txt

## Use
Start by downloading the FollowupQ dataset. TODO add link

Install the environment with the environment.yml or requirements.txt file

Put train.json, valid.json and test.json in the data map.

```bash
ENV_NAME="DL4NLP"
PYTHON_VERSION="3.10"

# Create a new Conda environment
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the new environment
echo "Activating the environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install the requirements from the requirements.txt file
echo "Installing requirements from requirements.txt..."
pip install -r ../requirements.txt # TODO change path of requirements
pip install transformers torch accelerate huggingface_hub datasets

# Done
echo "Conda environment '$ENV_NAME' created and packages installed."
```


Run `make_dataset.py` on all the splits, you can use the following bash script to do so:

```bash
srun python scripts/make_dataset.py \
    --model google/gemma-2-9b-it \
    --split train \ # Change to valid or test if needed
    --folder_path /data \ # Change to the path where the data is stored
    --seed 42 \
    --token $HF_TOKEN # Add your huggingface token
```

We made the perturbations on the training and validation set with `small_language_models.py`, so you can run the following script to generate the perturbed data:

```bash
MODEL=google/gemma-2-9b-it
MODEL=meta-llama/Llama-3.1-9B-Instruct
MASK_MODEL=T5-small
MASK_MODEL=T5-large
srun python scripts/small_language_models.py \
    --file_path FILE-PATH-WITH-LLM-ANSWERS-AND-FOLLOW-UPS \
    --model_name $MODEL \
    --mask_model $MASK_MODEL \
    --n_perturbations 5 \
    --detector_model $MODEL
```

```
meta-llama/Llama-3.1-9B-Instruct
```

```
meta-llama/Llama-3.1-70B-Instruct
```
```
google/gemma-2-9b-it
```
```
google/gemma-2-27b-it
```
