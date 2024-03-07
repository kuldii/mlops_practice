# pipeline.sh
#!/bin/bash

# Run requirements
pip install -r requirements.txt

# Run data_creation.py
python3 data_creation.py

# Run data_preprocessing.py
python3 data_preprocessing.py

# Run model_preparation.py
python3 model_preparation.py

# Run model_testing.py and print the accuracy
python3 model_testing.py
