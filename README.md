# Installing nHVAE
1. Install dependencies with the command: `pip install -r requirements.txt`
2. (Optional) Install ProGED for expression set generation: `pip install git+https://github.com/brencej/ProGED`

# Instructions for using nHVAE
1. Install nHVAE
2. Set up your config file (use _configs/config.json_ as a template). This file contains the needed parameters for training and evaluating the model
3. Create a set of expressions with the _expression_set_generation.py_ script and a suitable grammar (ProGED required) or use one of the existing data sets in the _data/expression_sets_ directory
4. Train an nHVAE model with the _train.py_ script
5. Evaluate the model using the _src/reconstruction_accuracy.py_ and _src/linear_interpolation.py_ scripts. For linear interpolation, the _data/expressions_ directory contains files with every expression in the existing data sets
