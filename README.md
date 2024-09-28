# Detecting Machine Generated Text vs. Human Generated Text

## Use
Start by downloading the FollowupQ dataset. TODO add link

Install the environment with the environment.yml or requirements.txt file

Put train.json, valid.json and test.json in the data map.

Run `make_dataset.py` on all the splits.

## Possible models
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