# Multi-Touch Attribution with LSTM-Attention Deep Learning Model
### Based on the paper "Deep Neural Net with Attention for Multi-channel Multi-touch Attribution" by Ning li, Sai Kumar Arava, Chen Dong, Zhenyu Yan, Abhishek Pani
implemented by Greg Murray

 https://arxiv.org/abs/1809.02230

## Foreword
The following is a description and physical implementation of the model based on the description in the paper.

## TLDR Results Summary
This notebook contains a prototype implementation of the deep learning model with attention weight vectors (**DLAW**), based on the 2018 Adobe paper in the link above, as a means of attributing channels' revenue from multi-touch impression data. The model performs much better on a simple, simulated dataset than the naive approaches 50-50 split, and first/last touched. 
 The data was simulated with an additive, mostly linear data generating function (see below for function and simulation methodology). The attribution percentage *mean absolute error* (MAE) for all approaches changed depending on the particular values chosen for the channel parameters in the simulated conversion logistic function (ie: the impact Facebook has as the second channel to impress, amount of boosting effect that a last-touched impression receives, etc.)   
 
 ### 95% Confidence Interval Absolute Error (AE), (*AE in range  [0, 1]*)

 ####  50/50 split test MAE range from 0.20-0.47 
 - (error depends entirely on how close the true attribution pct is to 50/50)
 
####  Last-Touched test MAE range from .60-.70
####  First-Touched test MAE range from .80-1
#### * **DLAW test MAE range from .10-.28**



## Simulation Methodology
### * Since the Li-Arava paper does not use a simulation or provide their data, and there was very minimal literature found on simulating MTA processes, a simulation had to be devised from scratch based on my own premises and assumptions.

The simulated data set included 100k observations with arbitrarily chosen beta coefficients and other parameters. The datset was split into 70k training obsbervations and 30k test observations. 

The data generating process (DGP) that leads to a conversion in this simple simulation is assumed to be a mostly linear process so as to make quantifying the contributions of each channel much easier, however more complex DGP's can and should be tested. To facilitate the the simulation and modeling effort, all observations were limited to exactly 2 impressions, where the channel in each time step was chosen from 3 possible channels (arbitrarily named Facebook, Google, and Snapchat). To simulate a conversion, a logistic equation is used wherein the contribution of each channel has a linear "strength" (beta) for each timestep. For example, if Facebook is seen in the first timestep then the effect is beta1_FB, and if it is seen in the second timestep then beta2_FB is used. In addition, the amount of time that elapses from the first timestep to the second decays exponentially weakening that channel's contribution to the conversion process. Also, the control variable of gender is added as an non-interactive term wherein a small additional amount is added to the conversion process independent on channel - this is less to do with what may be the case in reality and more as a means of inserting a control variable for the model to have to deal with. Lastly, relatively large, homoskedastic, Gaussian noise is added that frequently has a majority impact on the conversion outcome, in either the positive or negative direction. The intercept was set so as to balance the distribution of conversion probabilities to be more heavily massed around the low values (the default is to not convert except in cases of very large, positive noise terms). The intuition around the noise term being large is that there are many, highly-impactful, omitted variables that are mostly time-invariant, such as income and latent interest.

Since the DGP is additive relative to channel, the attribution of each channel can be quantified easily. The contribution for each channel is simply the sum of all the terms dependent on that channel (2 terms each), divided by the sum of the terms for all channels - which excludes the contribution from the noise terms. This gives us a distribution of channel contributions that can interpreted as the percent of revenue attributed to each channel (or in econometric terms, the ~variance of "y" explained by the channel's terms). Observations with conversion probabilities in the >90th percentile were assigned a conversion value of 1, all others were assigned 0.

## Data Generating Logistic Function
<p align="center">
  <img src="https://github.com/GregMurray30/MultiTouchAttribution/blob/main/logit_dgp.png">
</p>

### * It's important to note that the DL model makes no assumptions about the data generating function. This simulation DGP is used simply to generate data with some learnable signal that is partially dependent on the inputs, and from which the "true attributions" can be derived. 

### The generating functions for each of the parameters and inputs were as follows:
 
 - T1_FB, T1_GG, T1_SP, T2_FB, T2_GG, T2_SP ~ unif(1,3)
 - T1_elapse_time ~ gamma(2, 2)
 - gender ~ binomial(0.4)
 - v0 = -3
 - beta1_FB = 0.9
 - beta2_FB = 1.0, 1.8
 - beta1_GG = 1.7
 - beta2_GG = .8
 - beta1_SP = 0.4 , 1.4
 - beta2_SP = 0.5, 1.5
 - beta_lasttouch = .1, .2, .7, 1, 1.5, 2.2
 - beta_lgender = 0.4
 

In addition to its interpretability, the functional form of the model and particular parameter values were chosen to loosely simulate a plausible effect wherein Google is more impactful when it is the first impression because it implies higher latent interest (they googled some related terms independently) and the other two channels were either more or less impactful when seen last (depending on the value of beta_lasttouch) to simulate some higher or lower friction to the user purchase flow through their channel (ie: there is a better or worse funnel to purchase at FB than a Google ad).

I chose not to have a boosting effect in T2  for Google just so the model wouldn't be able to learn that T2 is always stronger. It may not be as plausible but it's more interesting to demonstrate that the attention model is actually learning to weight *contextualized* timestep vectors, not just the timesteps themselves.

## Model Description
A detailed description of the model can be found in the paper linked below but the intuition as to how a deep learning model, which are typically black box algorithms, can be used for feature inference is straightforward and outrageously elegant. A time-and-feature contextualized matrix, generated by feeding the touch-point matrix through an embedding and lstm layers, is passed to a multi-layer perceptron (attention layer) which then learns the corresponding weight that it should apply to each of the contextualized vectors of each time step. These weights are linearly interpretable so we can therefore simply add all the weights for a given channel if it appears at multiple touchpoints , or extract its weight as is from the corresponding time step weight if it appears only once. 

The key to understanding the power of this model is that because the attention layer receives a contextualized vector for each time step that has embedded in it a representation of the time-dependent, channel latent-features, and control variable features, it is not simply weighting the importance of each timestep independent of all the vital context - that would produce a rather useless model. Indeed, the model is incorporating all of the signal from the known information (gender, channel, timestep, elapsed time between time steps) to infer the proper amount of weight to give that channel-timestep vector towards predicting a conversion. Of course, it learns to shape these weights by its compulsion to minimize the binary crossentropy, guided by the labels themselves, without which this kind of learning would not be possible.

"Deep Neural Net with Attention for Multi-channel Multi-touch Attribution", Ning li, Sai Kumar Arava, Chen Dong, Zhenyu Yan, Abhishek Pani


## Model Evaluation
Once the model is trained, each input fed into the model will produce a different attention weight vector since that vector output is a function of the static, learned weights AND the transformed input ("attention weights" is a bit of a conflation of terms since they are not static, learned parameters).

The model performance asseessment using the simulations was not "publishably rigorous" as only a few different values for each of the parameters were tested and the distribubtions for the noise, proportion of each channel occurence, and control variable distributions were kept constant. For the attention weights to have any real meaning, however, it was important that the betas for each channel be disernibly different. Interestingly, the DLAW model performance appeared to be U-shaped as the beta that weighted the last-touched (beta_lasttouch) increased from 0 to >2.2. More specifically, DLAW model performance was around 14% MAE at value of beta_lasttouch of 0.2, then got worse to around 19% when beta_lasttouch was 0.7, then went back down to ~12% when beta_lasttouch was 2.2. I believe this shortcoming could be amended by adjusting the architecture or hyperparameters of the model in future iterations.

# Conclusions and Next Steps
### CONCLUSION
This was a very quick and preliminary exploration into the potential of this kind of approach to MTA modeling. The model's performance compared to the naive first/last touched methods was dominant. The 50/50 split model was really used as a marker to indicate when the actual simulated contributions were close to 50/50 so as to ensure the DLAW model can perform well regardless of the underlying contribution distribution. The impressive performance of the DLAW model (in this limited simulation) is therefore its ability to perform well in either scenario, regardless of whether the true attribution was close to 50/50 or not. 
However, to more fully support its viability it would need much more testing in terms of DGP functions, DGP parameter values, noise structure and magnitude, etc. In spite of the sub-rigorous testing, taking into account the success of the authors in implementing this model at Adobe, as well as the viability of the simulated data generating function as, at minimum, a plausible real world DGP, its performance definitely provides sufficient proof to warrant further investigation. 

### NEXT STEPS
As implemented, the model is not to the exact specifications of the model in the paper so perhaps incorporating hierarchical attention and a couple other small differences would help performance even further. Additionally, there may be unintended errors in the implementation of the model in this notebook, so it should be verified whether that is the case. In addition, the data processing portion of the notebook is not scalable to more channels/timesteps/etc, which is an engineering related improvement. Lastly, more robust models should be explored - as was done in the paper - so as to compare how this model fares against stiffer competition than naive models.
https://arxiv.org/abs/1809.02230
