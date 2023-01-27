# Methods


## Data split

<div align="justify">The test and validation sets each include 10% of the total number of days of data available, randomly sampled. The training set includes the remaining days. In the original dataset, there were many jobs that were included in more than one day/snapshot simply due to the fact that they stayed in the queue for more than 24h. This could result in the same job being included in both the train and test set, so we had to pre-process the data to avoid such situations. The end result is that a given job will now be found only in the snapshot of the day when the job was submitted.
<br></br>
This also protects us from another bad situation that could occur if we had a user submit 100 identical jobs in a batch and those jobs ended up being distributed across all train/valid/test sets. A simple prediction algorithm can be devised in such a way by making predictions for the queue time of a job by comparing it with other jobs in the training set submitted at the same time (thus probably in the same batch of many identical jobs). This has to be avoided because it does not lead to a proper system that could be deployed in practice. 
<br></br>
After taking all these precautions, we still have a decision to make about splitting the days chronologically or randomly amongst train/valid/test sets. We opted for the second method because, given our limited data, it diminished the overall variance in the project. In actual use in production we would train on all the past data and make predictions for today's jobs. The performance on "today's jobs" changes based on whether a day contains jobs that are easy or hard to predict. For this project, if we took "today" as the last day in our dataset, we would be at the mercy of the particular quirks from jobs of that day. This is one of the reasons why we instead decided to assign entire days to train/valid/test without following the chronological order (see <a href="3_Train_on_past_data">Training on Past Data</a>).
<br></br>
</div>

## Choice of models

<div align="justify">We started with a linear regression model to provide us with a good comparative baseline. This has the added advantage that it can be used to validate the correctness of our code because the exact solution to the least squares problem is known. We then trained neural networks consisting of a sequence of dense layers that all have the same number of hidden units and ReLU activation functions. By comparing them to the baseline we can assess whether there are gains made by using neural networks.
<br></br>
The following table shows the architecture and training hyperparameters of the neural networks selected. We have done a separate hyperparameter optimization sweep for the Cedar and Graham clusters because we wanted to train one model for each of the two data sources.
<br></br>
<div align="center">
<table>
  <tr>
   <td><strong>Hyperparameters</strong>
   </td>
   <td><strong>Cedar</strong>
   </td>
   <td><strong>Graham</strong>
   </td>
  </tr>
  <tr>
   <td>Number of layers
   </td>
   <td>7
   </td>
   <td>6
   </td>
  </tr>
  <tr>
   <td>Hidden size
   </td>
   <td>224
   </td>
   <td>72
   </td>
  </tr>
  <tr>
   <td>Learning rate
   </td>
   <td>0.001466
   </td>
   <td>0.001213
   </td>
  </tr>
  <tr>
   <td>L1 regularization
   </td>
   <td>0.0002983
   </td>
   <td>0.0001376
   </td>
  </tr>
  <tr>
   <td>L2 regularization coefficient
   </td>
   <td>0.004219
   </td>
   <td>0.0004128
   </td>
  </tr>
</table>
<i>Hyperparameters of neural networks selected for predicting queue time on Cedar and Graham clusters.
</i>
</div>
<br>
We proceeded to hyperparameters optimization (see file <code>sweep_random.yml</code>) using Weights & Biases taking as criterion the validation loss. In total, five hyperparameters were optimized: the learning rate, the number of layers, the number of hidden neurons per layer as well as the L1 and L2 regularization coefficients. Ten trials of training each model were performed to calculate the loss on the training, validation and test sets. Then, the predictions of the 10 trials were combined to obtain the distribution of the differences between the predictions and the actual wait time values.
<br></br>
It is worth noting that we did not go the route of modeling the data as time series in part because the nature of the SLURM jobs makes it hard to frame it like such, but also because the original data collection was set up for making predictions for each job in isolation. Indeed, the features of every job are augmented with extra features that represent the state of the cluster precisely so that predictions can be made without considering other external factors.
<br></br>

</div>

## Code Documentation

<div align="justify">To run a model training experiment, run the <code>run_experiment.py</code> script from the <code>slurm_queue_time_pred.wait_time_prediction</code> module specifying the desired training parameters and hyperparameters.
<br></br>
Here is the list of possible (hyper)parameters, as well as their default values:
</div>
<table>
 <tr>
  <td>--features
  </td>	 	
  <td>Comma-separated list of features
  </td>
  <td>Default: All features <i>(leave blank)</i>
  </td>
 </tr>
  <tr>
  <td>--model
  </td>	 	
  <td>The model to use for training
  </td>
  <td>Default: NN*
  </td>
 </tr>
 <tr>
  <td>-lr, --learning_rate
  </td>	 	
  <td>Learning rate
  </td>
  <td>Default: 1e-3
  </td>
 </tr>
 <tr>
  <td>--batch_size
  </td>	 	
  <td>Number of examples in one pass of the network
  </td>
  <td>Default: 128
  </td>
 </tr>
 <tr>
  <td>-o, --optimizer
  </td>	 	
  <td>The optimization function used to adjust parameters
  </td>
  <td>Default: adam**
  </td>
 </tr>
 <tr>
  <td>--hidden_size
  </td>	 	
  <td>Number of neurons in hidden layers
  </td>
  <td>Default: 128
  </td>
 </tr>
 <tr>
  <td>--nbr_layers
  </td>	 	
  <td>Number of hidden layers
  </td>
  <td>Default: 6
  </td>
 </tr>
 <tr>
  <td>--l1
  </td>	 	
  <td>L1 penalty coefficient (L1 regularization)
  </td>
  <td>Default: 0.0
  </td>
 </tr>
 <tr>
  <td>--l2
  </td>	 	
  <td>L2 penalty coefficient (L2 regularization)
  </td>
  <td>Default: 0.0
  </td>
 </tr>
 <tr>
  <td>--mix_train_valid
  </td>	 	
  <td>If present, examples in training and validation sets are from the same data pool
  </td>
  <td>
  </td>
 </tr>
 <tr>
  <td>-c, --cluster
  </td>	 	
  <td>Cluster from which to retrieve data
  </td>
  <td>Default: cedar***
  </td>
 </tr>
</table>

<div align="justify">
* Corresponds to a multi-layered neural network. The other valid possibility is linear for the linear regression model.

** Two values are allowed: adam or sgd.

*** Two values are allowed: cedar or graham.
<br><br>
Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/slurm_queue_time_pred/wait_time_prediction/run_experiment.py \
    --features=eligible_time,submit_time --model=linear \
    --learning_rate=0.00001 --batch_size=64 --optimizer=sgd \
    --hidden_size=64 --nbr_layers=3 --l1=0.0001 --l2=0.0001 --mix_train_valid --cluster=graham
```
