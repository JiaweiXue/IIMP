# IIMP: Individual Irregular Mobility Prediction
A bipartite graph neural network that predicts individual **irregular trips** to point-of-interests (POIs) one day in advance.

**Irregular trips**: the trips to destintations excluding homes and workplaces, encompassing venues like cinemas, parks, and shopping malls.

## Introduction

* IIMP provides forecasts for human travel to destinations outside of residences and workplaces, and these forecasts have various practical applications such as recommendation systems.
* In order to address the challenge of IIMP, we propose a web search-driven bipartite graph neural network, namely WS-BiGNN.
* WS-BiGNN utilizes historical mobility and web search records as input, with the goal of accurately predicting future human mobility.
* To effectively harness the insights gained from web search data, WS-BiGNN encompasses three specific design elements: hyperedges, a temporal weighting mechanism, and search mobility memory.
* Through experimental analysis using actual data from Yahoo! Japan Corportation under strict privacy protection rules, we demonstrate the valuable role of web search data and the superior prediction performance of WS-BiGNN.


## Directory structure
* **utils**: formulate the IIMP as a link prediction task in a bipartite graph, and prepare mobility feature and web search feature.  
* **model**: build the WS-BiGNN class.
* **result**: the screenshot of implementation results.
* **figure**: the problem and method.

## Requirements
* Python 2.7.5 or higher
* Torch 2.0.0 or higher 

## Manuscript
**Predicting Individual Irregular Mobility via Web Search-Driven Bipartite Graph Neural Networks.**
Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma\*, Satish V. Ukkusuri\*, June 2023.

## Utilizing past mobility and web search to predict future mobility

<p align="center">
  <img src="https://github.com/JiaweiXue/IIMP/blob/main/figure/fig_1_example.png" width="500">
</p>

## Formulate the IIMP problem as the link prediction task in bipartite graphs

<p align="center">
  <img src="https://github.com/JiaweiXue/IIMP/blob/main/figure/fig_2_network.png" width="350">
</p>

## Devise the search mobility memory module to model four interpretable patterns: Search&Go, Search&NotGo, NotSearch&Go, and NotSearch&NotGo

<p align="center">
  <img src="https://github.com/JiaweiXue/IIMP/blob/main/figure/fig_5_search_mobility.png" width="380">
</p>

## License
MIT license
