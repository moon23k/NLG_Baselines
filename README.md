# NLG_Basics
This repo covers Basic Architectures for Neural Language Generation Tasks (Neural Machine Translation and Dialogue Generation).
Each is **Sequence to Sequence**, **Attention Mechanism**, and **Transformer**.
The main purpose is to check the developments while comparing each model.
For a fairer comparision, some modifications are applied and as a result, some parts may differ from those in papers.

<br>


## Architecture desc

### [Sequence-to-Sequence](https://arxiv.org/abs/1409.3215)
As the name **"Sequence-to-Sequence"** suggests, it is an end-to-end sequence model.
The Architecture consists of Encoder and Decoder. In detail, the Encoder first makes Contetx Vectors from Input Sequences. 
And then the Decoder gets Encoder Outputs and Auto Regressive Values from Target sequences as an Input Values to return Target Sequences.
Before Sequence-to-Sequence Architecture was generally applied to various NLP Tasks, Statistical Models outperformed Neural End-to-End models.
This Architecture has proved its significance by opening Neural end-to-end Model Era.

<br>

### [Attention Mechanism](https://arxiv.org/abs/1409.0473)
The main idea of Attention Mechanism came from Human's Brain Cognition Process.
People live with a variety of information, but when faced with a specific problem, people usually focus on the information needed to solve the problem. We call this as an **Attention**.
The Architecture also use Encoder-Decoder architecture, but the difference is that the Decoder uses Attention Operation to make predictions.
By using Attention Mechanism, the model could avoid Bottle Neck problem, which results in Better performances in Quantative and Qualitive Evaluation at the same time.

<br>


### [Transformer](https://arxiv.org/abs/1706.03762)
Natural Language is inevitably a time-series data. In order to consider the time series aspect, the RNN structure was considered as the only option.
But **Transformer** broke this conventional prejudice and showed remarkable achievements by only using Attention Mechanism without any RNN Layer.
Existing RNN models always had two chronic problems. First is a vanishing gradient problem which is apparent as the sequence length gets longer. Second is Recurrent Operation process itself, which makes parallel processing difficult.
But the Transformer solved these problems only with Attentions. As a result, the architecture not only performs well in a variety of NLP tasks, but is also fast in speed.

<br>

### Model Configs
<table>
  <tr>
    <th>Seq2Seq</th>
    <th>Attention</th>
    <th>Transformer</th>
  </tr>
  <tr>
    <td><strong> &centerdot; input dimension:</strong> 10000 <br> <strong> &centerdot; output dimension:</strong> 10000 <br> <strong> &centerdot; embedding dimension:</strong> 256 <br> <strong> &centerdot; hidden dimension:</strong> 512 <br> <strong> &centerdot; n_layers:</strong> 2 <br> <strong> &centerdot; dropout ratio:</strong> 0.5</td>
    <td><strong> &centerdot; input dimension:</strong> 10000 <br> <strong> &centerdot; output dimension:</strong> 10000 <br> <strong> &centerdot; embedding dimension:</strong> 256 <br> <strong> &centerdot; hidden dimension:</strong> 512 <br> <strong> &centerdot; dropout ratio:</strong> 0.5</td>
    <td><strong> &centerdot; input dimension:</strong> 10000 <br> <strong> &centerdot; output dimension:</strong> 10000 <br> <strong> &centerdot; embedding dimension:</strong> 256 <br> <strong> &centerdot; hidden dimension:</strong> 256 <br> <strong> &centerdot; pff dimension:</strong> 512 <br> <strong> &centerdot; n_layers:</strong> 3 <br> <strong> &centerdot; n_heads:</strong> 8 <br> <strong> &centerdot; dropout ratio:</strong> 0.1 </td>
  </tr>
</table>

<br>
<br>

## Training Setup

* **Tasks:** Neural Machine Translation, Dialogue Generation
* **NMT Dataset:** downsized WMT14 EN-DE dataset (4.5M -> 450K)
* **Dialogue Dataset:** Daliy Dialgue Dataset (Processed from Multi-Turn to Single-Turn Dialogue)
* **Tokenization:** Applied Moses Tokenization first, and then applied BPE Tokenziation
* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam Optimizer
* **Label Smoothing:** 0.1
* **Learning Rate:** 1e-4 (LR Scheduler not applied)
* **Batch Size:** 128
* **Num of Epochs:** 5
* Applied Different Initialization for Each Models

<br>



<center>
  <img src="https://user-images.githubusercontent.com/71929682/168110116-374d3ac9-48d6-41e3-a2ce-d216f2e76422.png" width="70%" height="60%">
</center>


<br>
<br>

## How to Use
**First clone git repo in your env**
```
git clone https://github.com/moon23k/NMT_Basic
```

<br>

**Download and Process Dataset by the code below**
```
cd NMT_Basic
bash prepare_data.sh
```

<br>

**Train models with "train.py" file (scheduler is optional)**
```
python3 train.py -model ['seq2seq', 'attention', 'transformer'] -task ['translate', 'dialogue'] -scheduler ['constant', 'noam', 'cosine_annealing_warm', 'exponential', 'step']
```

<br>

**Test trained models with "test.py" file**
```
python3 test.py -model ['seq2seq', 'attention', 'transformer'] -task ['translate', 'dialogue']
```

<br>

**Test with user input sentence via trained models**
```
python3 inference.py -model ['seq2seq', 'attention', 'transformer'] -task ['translate', 'dialogue']
```


<br>
<br>


## Results

### NMT Task
**Expected BLEU Score** 
* Seq2Seq Model : About 10
* Seq2Seq with Attention Model : About 15
* Transformer Model : About 20

(The value based on the Best Performance posed on "paperswithcode" home page with wmt14 en-de dataset)

<br>

**Actual BLEU Score**
* Seq2Seq Model : 
* Seq2Seq with Attention Model : 
* Transformer Model : 

<br>

### Dialogue Generation Task

<br>
<br>

## Reference
* Sequence to Sequence Learning with Neural Networks
* Neural Machine Translation by Jointly Learning to Align and Translate
* Attention is all you need
