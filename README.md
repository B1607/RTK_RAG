# RTK_RAG: Leveraging Retrieval Augmented Generation with Multi-Window Convolutional Neural Networks for Superior ATP Binding Site Prediction in Receptor Tyrosine Kinases
Sin-Siang Wei, Wei-En Jhang, Yu-Chen Liu, Cheng-Che Chuang,  Yu-Yen Ou
|[ 🎇&nbsp;Abstract](#abstract) |[📃&nbsp;Dataset](#Dataset) |[ 🚀&nbsp;Quick Prediction ](#colab)|[ 📚&nbsp;License](#License)|
|-------------------------------|-----------------------------|------------------------------------|-----------------------------|

## 🎇Abstract <a name="abstract"></a>
Receptor Tyrosine Kinases (RTKs) are key regulators of cellular signaling and are frequently involved in cancer development. As their activation depends on ATP binding to the kinase domain, precisely identifying ATP binding sites is critical for mechanistic studies and targeted therapy development. However, general ATP binding site prediction methods often fall short for RTKs due to their diverse structural features across different protein families. To address this challenge, we introduce RTK_RAG, a framework that integrates retrieval-augmented generation (RAG) and utilizes protein language models (PLMs) with a multi-window convolutional neural network (MCNN) architecture to improve ATP binding site prediction for RTKs. When tested on an independent RTK dataset, RTK_RAG outperforms general ATP binding site predictors on multiple evaluation metrics. By accounting for RTK-specific structural differences, our study provides a reliable tool for researching RTK function and facilitating the development of novel kinase inhibitors. Moreover, this approach demonstrates the potential of RAG-based frameworks for enhancing functional predictions in specialized protein families, offering a generalizable strategy for improving binding site identification in specific protein families.
<br>

![workflow](https://github.com/B1607/RTK_RAG/blob/3cd56468802ae8d70bcb21f62606895eb7357b0d/Figures/RTKs-Workflow.png)

## 📃Dataset <a name="Dataset"></a>

| Dataset            | Protein Sequence |    ATP Binding Residues  | Non-Binding Residues    |
|--------------------|------------------|--------------------------|--------------------------|
| ATP-549 (Train)    |   549             |          8,502             |205,035                      |
| RTKs (Independent Test)     |    69         |                    711    |     73,270                  |
| RAG database       |          244  |                     2,676   |                 261,088      |

## 🚀Quick Prediction <a name="colab"></a>
[<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/drive/19pOze_jxxAq3XOQd6yuIiCBLEpbGBI29?usp=sharing)<br>
https://colab.research.google.com/drive/19pOze_jxxAq3XOQd6yuIiCBLEpbGBI29?usp=sharing

### Step 1: Environment Setup
Open the link of Google colab and change the hardware to a CPU device(recommended).

### Step 2: Excute the program
This Colab notebook will automatically import all necessary dependencies and download the required files.

### Step 3: Submit your fasta file and wait for the Prediction result 

Upload your FASTA file to run the prediction.
The format of the FASTA file will be as follows:
```bash
>Q9C7T7
MSRRPDLLRGSVVATVAATFLLFIFPPNVESTVEKQALFRFKNRLDDSHNILQSWKPSDSPCVFRGIT
>"Name"
(Animo Acid Sequence)
```
(Or use our dataset for RTK testing. [⬇️link](https://github.com/B1607/RTK_RAG/tree/main/Data/Fasta/RTK_Test)<br>
The result will be formatted as follows:
```bash
Fasta     :  >Q9C7T7
Amino acid:  MSRRPDLLRGSVVATVAATFLLFIFPPNVESTVEKQALFRFKNRLDDSHNILQSWKPSDSPCVFRGIT...
Prediction:  1000000000000000000000000000000000000000000000000000000000000000000...
```
1 indicates the amino acid is predicted to be a ATP-Binding residue.<br>
0 indicates the amino acid is predicted to be a non-ATP-Binding residue.

## 📚&nbsp;License <a name="License"></a>
Licensed under the Academic Free License version 3.0
