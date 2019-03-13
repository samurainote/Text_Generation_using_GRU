# GRU: Text_Generation
## GRUリカーレントネットワークを用いた文章生成
![](https://camo.githubusercontent.com/9a5b885799c2d8e50f3f049fde2ada7696e974ca/68747470733a2f2f692e696d6775722e636f6d2f484646575674432e706e673f32)

## Introduction

What is GRU()?

The Gated Recurrent Unit (GRU) is a model that makes LSTM a little simpler. The input gate and the forgetting gate are integrated into one gate as an "update gate".   
Similar to LSTM, introducing such an oblivion and update gate makes it easy to maintain the memory of the features of events before long steps.    
That's because it can be said that shortcut paths that bypass between each time step are efficiently generated.     
Because of this, errors can be easily back-propagated during learning, which reduces the problem of gradient loss.       

When GRU become "a better choice" than LSTMs?

The biggest difference between GRU and LSTM is that GRU is faster and easier to execute (but it is not rich in expressiveness).    
In practice, the advantages tend to offset the weaknesses, such as at the expense of performance, as large networks are needed, for example, to enrich the expressive power. GRU performs better than LSTM when it does not require expressive power.     

## Technical Preferences

| Title | Detail |
|:-----------:|:------------------------------------------------|
| Environment | MacOS Mojave 10.14.3 |
| Language | Python |
| Library | Kras, scikit-learn, Numpy, matplotlib, Pandas, Seaborn |
| Dataset | [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset) |
| Algorithm | GRU Network |

## Refference

- [Understanding GRU Networks](https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be)
- [Creating A Text Generator Using Recurrent Neural Network](https://chunml.github.io/ChunML.github.io/project/Creating-Text-Generator-Using-Recurrent-Neural-Network/)
- [Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
- [An applied introduction to LSTMs for text generation — using Keras and GPU-enabled Kaggle Kernels](https://medium.freecodecamp.org/applied-introduction-to-lstms-for-text-generation-380158b29fb3)
- [Word-level LSTM text generator. Creating automatic song lyrics with Neural Networks](https://medium.com/coinmonks/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)
- [ニューラルネットワークの動物園 : ニューラルネットワーク・アーキテクチャのチートシート(後編)](https://postd.cc/neural-network-zoo-latter/)
