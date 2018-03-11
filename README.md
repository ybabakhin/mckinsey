# McKinsey Hackathon
It's a second place solution for McKinsey Analytics Hackathon:

https://datahack.analyticsvidhya.com/contest/mckinsey-analytics-online-hackathon-recommendation

## Problem statement
The client has provided you with history of last 10 challenges the user has solved, and you need to predict which might be the next 3 challenges the user might be interested to solve

## Solution
For each user in the training set create 3 observations: sequence of 10 challenges solved and three labels (for 11th, 12th and 13th challenges). So, now we have a multiclassification problem with 5k classes and about 200k observations. The classification is done using a single Recurrent Neural Net with BiDirectional LSTM layer.

During test time, obtain a probability distribution for each sequence. Then, choose top-3 argmax probabilities as 11th, 12th and 13th challenges predicted.
