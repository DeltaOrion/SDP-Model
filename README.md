# SDP Model

SDP Model and experimental setup for FYP Paper. 

## Dataset

 * Uses the BugHunter Dataset. 
 * Follow preprocessing instructions [here](https://github.com/DeltaOrion/BugHunter-Preprocessing)
 * Store the results in a folder named `dataset` or edit the scripts to point to the right location of the dataset

## Usage

 * Select the model `cd src/sdpt5-simple/nn/`
 * Train the model using `python run.py`
 * Evaluate using `python evaluate.py`
 * Predict on unseen data with `python predict.py`
