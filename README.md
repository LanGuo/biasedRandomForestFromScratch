
This is an implementation from scratch (using numpy, pandas, scipy) of decision tree (for classification), random forest, sampling and preprocessing methods, as well as model performance metrics and their visualization (matplotlib).

Also included an implementation of Biased Random Forest (BRAF) as outlined in the paper, “Biased Random Forest for Dealing with the Class Imbalance Problem”. This is an algorithm designed to or combating the class imbalance problem in classification at the algorithm-level rather than the data-level.

To test:
Option 1 - python virtual env on Linux:
1. Clone this repo.
2. Make sure you have permission to execute run.sh, if not, run `chmod u+x run.sh`
3. Run `./run.sh`, this will make a virtual env, install dependencies, then run the python script that tests the BRAF implementation and print out stuff in the terminal as well as save some figures

Option 2 - without installing dependencies (assuming your system already has python3 and the packages in requirements.txt installed):
1. Clone this repo.
2. Run ```python test_rf.py --critical_ratio 0.5 --k_nearest 10 \
--n_estimators 100 --max_sample_frac 0.6 --max_depth 6 \
--min_samples_leaf 4 --max_features 6 \
--output_dir cv_output --n_process 4```