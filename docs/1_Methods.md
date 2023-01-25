# Methods


## Data split

<div style="text-align: justify">The test and validation sets each include 10% of the total number of days of data available, randomly sampled. The training set includes the remaining days. There is no overlap between data from different sets. Also, samples (jobs) belonging to the same day are part of the same set. We have chosen to separate by day rather than by job to avoid the absurd situation where we would share, between the training, validation and test sets, identical jobs submitted by the same user at the same time. This way of splitting the data would in this case lead to better learning by the model.
<br></br>
We had the choice to split the days chronologically or randomly. We opted for the second method, since it reduces the impact on the predictive performance of the variability that exists between jobs in the training and test sets. However, in the operating environment, training models with data from previous days is more realistic (see <a href="docs/3_Train_on_past_data">Training on Past Data</a>).
<br></br>
</div>

## Choice of models

<div style="text-align: justify">The models used to predict the waiting time of jobs on computing clusters are described here. We first implemented a linear regression model. The advantage of this model is that it is simple to implement and that it makes it possible to verify that the inputs and outputs of the system are correct. Additionally, it is possible to calculate the exact solution to the linear problem using least squares, since the mean square error (MSE) or loss effectively corresponds to the error rate. This provides a benchmark for the minimum loss expected after training.
<br></br>
The following table presents the parameters of the deep neural network variants (i.e. with several hidden layers) chosen for Cedar and Graham clusters (for which training data is available).
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
<i>Hyperparameters of deep neural networks selected for predicting queue time on Cedar and Graham clusters.
</i>
</div>
<br>
These neural networks possess ReLU activations between layers. The use of attention and transformer models has been discarded because the data is non-sequential in nature.
<br></br>
We proceeded to hyperparameters optimization using Weights & Biases taking as criterion the validation loss. In total, five hyperparameters were optimized: the learning rate, the number of layers, the number of hidden neurons per layer as well as the L1 and L2 regularization coefficients. Ten trials of training each model were performed to calculate the loss on the training, validation and test sets. Then, the predictions of the 10 trials were combined to obtain the distribution of the differences between the predictions and the actual wait time values.
<br></br>
</div>

## Code Documentation

<div style="text-align: justify">To run a model training experiment, run the <b>run_experiment.py</b> script from the <b>code.wait_time_prediction</b> module specifying the desired training parameters and hyperparameters.
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

<div style="text-align: justify">
* Corresponds to a multi-layered deep neural network. The other valid possibility is linear for the linear regression model.

** Two values ​​are allowed: adam or sgd.

*** Two values ​​are allowed: cedar or graham.
<br><br>
Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/code/wait_time_prediction/run_experiment.py --features=eligible_time,submit_time --model=linear -- learning_rate=0.00001 --batch_size=64 --optimizer=sgd --hidden_size=64 --nbr_layers=3 --l1=0.0001 --l2=0.0001 --mix_train_valid --cluster=graham
```