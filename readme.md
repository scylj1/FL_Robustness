Robustness of Federated Learning
========
A repository to evaluate the robustness of federated learning (FL) in natural language inference (NLI) task, created by Lekang Jiang. 

`fed.py` is the main function to start FL, which contains federated clients definitions. 

`myutils.py` includes helper functions, dataset partitioning, model initialisation, training and evaluating functions. 

`freeze.py` contains functions to freeze some layers in the model. 

`data/preprocess.py` is used to download datasets and pre-process them for training. 

To run the code, follow these instructions. 


Setup
-----
1. Create python environment (e.g. `conda create -n fed` and `conda activate fed`)

2. Run the install: `pip install -r requirements.txt`


Prepare dataset
-------------------
1. Specify the hyper-parameter in `data/preprocess.sh`, especially the `--output_dir`.
2. To create label distribution skew, set `--do_noniid` to True and adjust `--alpha` and `--beta` to control the non-IIDness. 
3. Run: `bash data/preprocess.sh`

## Run

1. Specify the hyper-parameter in `run_fed.sh`. Set the `--fed_dir_data` to the path where you prepare the datasets. 
2. To create quantity skew, set `--do_noniid` to True and adjust `--alpha` and `--beta` to control the non-IIDness.
3. To adopt layer freezing method, set `--do_freeze` to True and adjust `--num_freeze_layers`. 
4. Run: `bash run_fed.sh`


Acknowledgement
-----------
Thanks for the example code of federated learning on [Flower](https://github.com/adap/flower) and natural language processing on [Huggingface](https://github.com/huggingface/transformers). 