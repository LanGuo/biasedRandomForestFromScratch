#!/usr/bin/env bash
# This is the script to run when testing this code

# make a python3 virtual environment for testing
python3 -m venv braf
cd braf
source bin/activate

# go back to LG_technical which contains the code
cd ..
# install requirements
pip install -r ./requirements.txt

# run main script
python ./test_rf.py --critical_ratio 0.5 --k_nearest 10 \
--n_estimators 100 --max_sample_frac 0.6 --max_depth 6 \
--min_samples_leaf 4 --max_features 6 \
--output_dir cv_output --n_process 4