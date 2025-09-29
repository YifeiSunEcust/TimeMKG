# TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling
The repo is the official implementation for the paper: [TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling](https://arxiv.org/pdf/2508.09630).

## Methods
The overall structure of TimeMKG consists of four key modules: multivariate knowledge graph, dual-modality encoder, cross-modality attention, and inference module.
![TimeMKG](https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/TimeMKG.png)  

## 🛠️Setup
Install Python 3.8. For convenience, execute the following command.
```bash
pip install -r requirements.txt
```
- Some datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view). 
- Then place the downloaded data in the folder `./dataset`. Here is a summary of supported datasets.
- If you use a custom dataset, please store the data as a `.csv `file with the following structure, using `Dataset_Custom` in `data_loader.py` uses a default split of 7:2:1 for training set:test set:validation set.
<p align="center">
  <img src="https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/data.png" alt="csv">
</p>

- Add the prior knowledge required for the dataset to the `./dataset_txt` folder. An example of causal relationship description is shown in `ETT.txt`.

```
# Electricity Transformer Temperature (ETT) Knowledge Graph
# Variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

[Entities]
# Power load features
HUFL | description | High Voltage Phase Load
HULL | description | High Voltage Line Load
MUFL | description | Medium Voltage Phase Load
MULL | description | Medium Voltage Line Load
LUFL | description | Low Voltage Phase Load
LULL | description | Low Voltage Line Load
# Target variable
OT | description | Oil Temperature
[Relationships]
# Intra-voltage level relations
HUFL -> HULL | relation | Phase-line correlation within same voltage level
MUFL -> MULL | relation | Phase-line correlation within same voltage level
LUFL -> LULL | relation | Phase-line correlation within same voltage level
...
```

## MKG Construction
### Input: knowledge text ➡️  knowledge graph + prompts
Use `MKG_generate.py` to build the Multivariate Knowledge Graph and generate prompt. 
Modify the configuration of MKG_generate according to the dataset.
```
    txt_path = "./dataset_txt/ETT.txt"
    variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'] 
    output_file = "./prompts/ETT_prompt.txt" 
    MKG(txt_path, variables, output_file)
```
The structure of the generated prompt should be: `HUFL: High loads may cause temperature rise.`

```bash
python MKG_generate.py
```
MKG is built based on [LightRAG](https://github.com/HKUDS/LightRAG). 

## TimeMKG
### Input: prompt+seq ➡️  seq
TimeMKG and baselines are in the `./model`. Depending on the seq2seq task, you can choose `Long_term_forecasting`, `Short_term_forecasting` and `Classification`.
```bash
python run.py
```
We provide a more detailed and complete command description for training and testing the model:
An example config on `ETTh1` as custom dataset.
# basic config
```python
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model', type=str, default='TimeMKG')

    # data loader
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='', help='data file')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--prompt_path', type=str, default='./prompt/ETT_prompts')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--max_len', type=int, default=30, help='max length of CPEncoder')
```

## MKG example:
![ETT_MKG](https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/Graphprompt.png)  
