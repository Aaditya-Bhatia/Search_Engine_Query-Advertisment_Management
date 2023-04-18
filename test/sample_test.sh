#!/bin/bash

# Run first Python file with arguments
python 0_validate_labeled_data.py -l reference_data.csv

# Run second Python file with arguments
python 1_build_files.py -d Data/ -l reference_data.csv

# Run third Python file with arguments
python 3a_build_aggregate_model.py 

# Run fourth Python file with arguments
python 3b_build_specific_modes.py

# for the retraining, (incase we get more reference labeled data)
python 3c_retrain.py

# to get the predictions from data
python 4_get_predictions.py -l reference_data.csv -x new_predictions.csv -d Modles/

