# RTK_RAG: Leveraging Retrieval Augmented Generation with Multi-Window Convolutional Neural Networks for Superior ATP Binding Site Prediction in Receptor Tyrosine Kinases

|[ 🎇&nbsp;Abstract](#abstract) |[📃&nbsp;Dataset](#Dataset) |[ 📚&nbsp;License](#License)|
|-------------------------------|-----------------------------|---------------------------------|

## 🎇Abstract <a name="abstract"></a>
Receptor Tyrosine Kinases (RTKs) are key regulators of cellular signaling and are frequently involved in cancer development. As their activation depends on ATP binding to the kinase domain, precisely identifying ATP binding sites is critical for mechanistic studies and targeted therapy development. However, general ATP binding site prediction methods often fall short for RTKs due to their diverse structural features across different protein families. To address this challenge, we introduce RTK_RAG, a framework that integrates retrieval-augmented generation (RAG) and utilizes protein language models (PLMs) with a multi-window convolutional neural network (MCNN) architecture to improve ATP binding site prediction for RTKs. When tested on an independent RTK dataset, RTK_RAG outperforms general ATP binding site predictors on multiple evaluation metrics. By accounting for RTK-specific structural differences, our study provides a reliable tool for researching RTK function and facilitating the development of novel kinase inhibitors. Moreover, this approach demonstrates the potential of RAG-based frameworks for enhancing functional predictions in specialized protein families, offering a generalizable strategy for improving binding site identification in specific protein families.
<br>

![workflow]()

## 📃Dataset <a name="Dataset"></a>

| Dataset            | Protein Sequence |    ATP Binding Residues  | Non-Binding Residues    |
|--------------------|------------------|--------------------------|--------------------------|
| ATP-549 (Train)    |   549             |          8,502             |205,035                      |
| RTKs (Independent Test)     |    69         |                    711    |     73,270                  |
| RAG database       |          244  |                     2,676   |                 261,088      |

## 📚&nbsp;License <a name="License"></a>
Licensed under the Academic Free License version 3.0
