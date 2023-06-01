# IIMP: Individual Irregular Mobility Prediction
A bipartite graph neural network forecasting one-day ahead individual irregular trips to point-of-interests (POIs).

Regular trips: the journeys made to homes or workplaces.

Irregular trips: the trips to destintations other than homes and workplaces, such as cinemas, parks, and shopping malls.

## Introduction

* IIMP provides forecasts for human travel to destinations outside of residences and workplaces, and these forecasts have various practical applications such as recommendation systems.
* In order to address the challenge of IIMP, we propose a web search-driven bipartite graph neural network, namely WS-BiGNN.
* WS-BiGNN utilizes historical mobility and web search records as input, with the goal of accurately predicting future human mobility.
* To effectively harness the insights gained from web search data, WS-BiGNN encompasses three specific design elements: hyperedges, a temporal weighting mechanism, and search mobility memory.
* Through experimental analysis using actual data from Yahoo! Japan Corportation under strict privacy protection rules, we demonstrate the valuable role of web search data and the superior prediction performance of WS-BiGNN.


**Predicting Individual Irregular Mobility via Web Search-Driven Bipartite Graph Neural Networks.**
Jiawei Xue, Takahiro Yabe, Kota Tsubouchi, Jianzhu Ma\*, Satish V. Ukkusuri\*, June 2023.

## Requirements
* Python 2.7.5 or higher
* Torch 2.0.0 or higher 

## Directory Structure
* **utils**: formulate the IIMP as a link prediction task on a bipartite graph, and prepare mobility feature and web search feature.  
* **model**: build the WS-BiGNN class.
* **result**: the screenshot of implementation results.
* **figure**: the problem and method.

## License
MIT license
