# TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling
The repo is the official implementation for the paper: [TimeMKG: Knowledge-Infused Causal Reasoning for Multivariate Time Series Modeling](https://arxiv.org/pdf/2508.09630).

## Introduction
Multivariate time series data has two main parts: variable semantics (like variable names and descriptions) and sampled numerical observations, but traditional time series models treat variables as anonymous statistical signals and ignore the useful domain knowledge in those textual descriptors—knowledge that’s important for strong, interpretable models. To fix this, we present TimeMKG, a multimodal causal reasoning framework that moves time series modeling from low-level signal processing to knowledge-informed inference: it uses large language models to understand variable semantics, builds structured Multivariate Knowledge Graphs to capture relationships between variables.

![Introduction](https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/Introduction.png)  

## Methods
The overall structure of TimeMKG consists of four key modules: multivariate knowledge graph, dual-modality encoder, cross-modality attention, and inference module.
![TimeMKG](https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/TimeMKG.png)  

## Usage
### MKG Construction: IN: Text OUT: MKG+prompt
Add the prior knowledge required for the dataset to the dataset_txt folder. An example of causal relationship description is shown in "ETT.txt".
Use MKG_generate.py to build the Multivariate Knowledge Graph and generate prompt. Modify the configuration of MKG_generate according to the dataset.
```bash
python MKG_generate.py
```
Prompt example:
"HUFL:High voltage load changes may affect medium voltage system."

MKG is built based on [LightRAG](https://github.com/HKUDS/LightRAG). The configuration and modification methods of LLM can refer to lightrag.

### Dadasets
Some datasets can be obtained from [Google Drive](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view). The others can be obtained from the links in Appendix.

### baseline: IN:seq OUT:seq
TimeMKG and baselines are in the model. Depending on the seq2seq task, you can choose Long_term_forecasting, Short_term_forecasting and Classification.

### TimeMKG: IN:seq+prompt OUT:seq
```bash
python run.py
```

## MKG example:
![ETT_MKG](https://github.com/YifeiSunEcust/TimeMKG/blob/main/fig/Graphprompt.png)  
