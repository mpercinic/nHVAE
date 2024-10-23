# Installing nHVAE
1. Install rust ([https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))
2. Create a new (conda) environment
3. Install dependencies with the command: `pip install -r requirements.txt`
4. (Optional) Install ProGED for expression set generation: `pip install git+https://github.com/brencej/ProGED`

# Instructions
1. Install nHVAE
2. Set up your config file (use _configs/test_config.json_ as a template)
3. Create a set of expressions with the _expression_set_generation.py_ script and a suitable grammar or use one of the existing data sets in the _data/expression_sets_ directory
4. Train an nHVAE model with the _train.py_ script
5. Evaluate the model using the _src/reconstruction_accuracy.py_ and _src/linear_interpolation.py_ scripts
