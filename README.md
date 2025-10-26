##  G2S-HCVAE: Diverse Retrosynthetic Planning via Hierarchical Latent Variables
## Reproduce the Results
### 1. Environmental setup
Please ensure that conda has been properly initialized, i.e. **conda activate** is runnable. 
```
conda create -n retrog2s python==3.9.19
conda activate retrog2s
pip install -r requirements.txt
```

### 2. Preprocess Reaction Data
Preprocess the csv file to batch. The original files can be manually downloaded via https://drive.google.com/drive/folders/1_uoz4SqiFro8mS8b1KHbGWp3ioQeS2o2?dmr=1&ec=wgc-drive-hero-goto.
```
python preprocess.py --dataset_name $DATASET
```
$DATASET can be selected from[**uspto_50k**, **uspto_diverse**, **uspto_full_ms**].

### 3. Model training and validation
#### 3.1 Taining
Here is the training command and descriptions for each argument:
```
export CUDA_VISIBLE_DEVICES=1 # set id of CUDA device, 0 in default

python train.py 

# data and model
--dataset_name          # Dataset                               [uspto_50k, uspto_diverse, uspto_full_ms]
--split_type            # SMILES tokenization                   [char, token]
--model_type            # model_name                            ['BiG2S', 'BiG2S_RXN_HCVAE', 'S2S', 'S2S_RXN_HCVAE']
--cvae_type             # forms of cvae                         ['dcvae']
--representation_form   # how to represent product and reactants in translation 
                                                                ['graph2smiles', 'smiles2smiles']  
# training pipelines
--train_task            # the prediction task to perform (forward reaction prediction/Retrosynthesis/both)    
                                                                ['prod2subs', 'subs2prod', 'bidirection']
--loss_type             # training loss                         ['CE', 'Mixed-CE', 'focal']   
--lat_disc_size         # number of latent reaction classes     [10, 30, 60, 90, 120]
--lat_z_size 256        # size of continuoues latent variables  [256]
--kl_anneal             # annealing of KL Loss
# inference
--beam_module           # beam_search_module                    ['OpenNMT', 'huggingface']    
                               
```
Optionally, one can run the demo command:  
```
python train.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 800 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --anneal_kl --beam_strategy OpenNMT
```
Please note that the model_names that contains `G2S` should be coupled with `graph2smiles` representation forms, and models containing `S2S` should be with `smiles2smiles` forms.

One can download the checkpoints to bypass the training process, via https://drive.google.com/drive/folders/1Y1RNCisnVfTBbJPB2VPDEhC6t5YDQ08i?dmr=1&ec=wgc-drive-hero-goto.

#### 3.2 Evaluation
Run the `predict_one_one.py` (for one-to-one mapped reaction data) or `predict_one_N.py` (for one-to-many mapped reaction data) with the same arguments in train command, for example:
```
python predict_one_N.py --dataset_name uspto_diverse --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 800 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --beam_strategy OpenNMT
```
The results of Top-k Acc, Coverage rate will be saved in the directory ’checkpoints‘, the information of predictions will be saved in directory 'results'.
#### 3.3 Round-Trip Experiment
One who want to perform round-trip experiment, should install [Mol Transformer](https://github.com/pschwllr/MolecularTransformer) previously.
After installation, run the command:
```
python round_trip.py --dataset_name uspto_diverse --result_file <copy the path wanted csv in directory `results`>
```
The results will be saved in directory 'round_trip'.

### 4. A* based Multi-step retrosynthetic planning
We adopt the code of [FusionRetro](https://github.com/SongtaoLiu0823/FusionRetro) to perform A* search algorithm to find synthetic routes from starting material to the target molecules. Thanks for Liu's contributions.  
**Note**: This procedure is implemented on the *Retro-Bench* dataset, which needs to train the single-step model before searching routes, one can run this lines of command to achieve it.  

For training: 
```
python train.py --dataset_name uspto_full_ms --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 200 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --anneal_kl 
```
For routes searching:
```
python rerank_star.py --dataset_name uspto_full_ms --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 800 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --anneal_kl --beam_strategy OpenNMT
```

The evaluation protocol shares the same code with the search process, with the only addition of one argument `--load_res`.

### 5. Extensions
#### 5.1 Inference for a Chosen Molecule
To perform retrosynthesis or forward reaction prediction for arbitrary molecules, one replace the string in `input_smi` int python file `inference.py` and run the command:  
```
python inference.py --dataset_name uspto_full_ms --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 800 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --anneal_kl --beam_strategy OpenNMT
```
#### 5.2 Visualization of latent reaction classes embeddings
```
python tsne_vis.py --dataset_name uspto_full_ms --model_type BiG2S_HCVAE_RXN --representation_form graph2smiles --beam_module OpenNMT --train_task bidirection --loss_type CE --lat_disc_size 10 --epochs 800 --lat_z_size 256 --split_type token --cvae_type dcvae --cvae_layers 6 --anneal_kl --beam_strategy OpenNMT
```

## Acknowledgement
We also refer to the codes of [Graph2SMILES](https://github.com/coleygroup/Graph2SMILES), [RetroDCVAE](https://github.com/MIRALab-USTC/DD-RetroDCVAE), [BiG2S](https://github.com/AILBC/BiG2S) and [RetroBridge](https://github.com/igashov/RetroBridge) during implementation. Thanks for their contributions.  
