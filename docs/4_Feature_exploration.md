# Feature exploration

<div align="justify">One of the goals of our collaborators at the DRAC was to know the most significant features for predicting the waiting time of a job, for explainability purposes and to guide users on the judicious choice of parameters for the <code>sbatch</code> and <code>salloc</code> commands when submitting jobs on compute clusters. Thus, we wanted to determine the features that best predict waiting time. That is, which small subset of features are the best minmizing the MSE?
<br></br>
We used a greedy approach to construct a set of features, iteratively adding features based on which one would allow for the most important improvement (in terms of lowest MSE on the training set). Starting from a empty set, we try each feature individually to pick out the one which can yield the lowest MSE. We add the best and we continue the process until we have selected the 10 best features. Every time we perform an evaluation, this involves training a linear model or a neural network for predictions.
<br></br>
</div>

## Results

<div align="justify">The following figures show the results of the feature exploration with the linear model and the 7-layer neural network model used to make predictions on Cedar data. For each of the models, the curves of the training and test loss (MSE) according to the cumulative features for the training are presented.
<br><br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/exploration_results/cedar_linear/plots/10_features_plot_training.png">
   </td>
   <td><img src="../results/exploration_results/cedar_linear/plots/10_features_plot_test.png">
   </td>
  </tr>
 </table>
 <i>Training (left) and test (right) loss according to cumulative features used for training the linear model, using data from Cedar.
 </i>
</div>
<br>
<br><br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/exploration_results/cedar_7NN/plots/10_features_plot_training.png">
   </td>
   <td><img src="../results/exploration_results/cedar_7NN/plots/10_features_plot_test.png">
   </td>
  </tr>
 </table>
 <i>Training (left) and test (right) loss according to cumulative features used for training the 7-layer neural network model, using data from Cedar.
 </i>
</div>
<br>
The green hatched lines in the graphs above represent the MSE obtained by training the model with the set of all features (76). Because a properly-trained model can always chose to ignore some input features (using a zero coefficient), we get a monotonic decrease in the loss by adding more features. This applies to the training loss, but not to the test loss.
<br></br>
We find that the features selected by each of the models and the order of selection vary. This can be explained by the varying complexity of the models used; from the linear model to the 7-layer model, the complexity increases. One of the visible trends is that all models tend to make the best use of the so-called "augmented" features. That is, features that were artificially added to represent the state of the visible SLURM queue with many jobs already scheduled. Those jobs are identified with a naming scheme that matches <code>*_above</code>.
<br></br>
Among these jobs, those pending with higher priority (<code>q_*_above</code>) seem to have a greater predictive importance than the running jobs (<code>r_*_above</code>). Overall, the amount of memory, GPUs and CPU cores used on the computing cluster when submitting the job appear to be good predictors of the job’s waiting time.
<br></br>
</div>

## Code Documentation

<div align="justify">To run a feature exploration experiment, run the <code>run_experiment.py</code> script from the <code>slurm_queue_time_pred.explore_features</code> module specifying the desired training (hyper)parameters, as listed in <a href="1_Methods.md"> Methods</a>. The features argument does not apply. Here are the additional arguments:
<br></br>
</div>
<table>
 <tr>
  <td>-n, --nbr_top_features
  </td>	 	
  <td>Number of cumulative features used for training
  </td>
  <td>Default: 10
  </td>
 </tr>
  <tr>
  <td>--output_dir
  </td>	 	
  <td>Directory name of the results.exploration_results directory  where the results JSON files will be written
  </td>
  <td>
  </td>
 </tr>
</table>
<br>
<div align="justify">Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/slurm_queue_time_pred/explore_features/run_experiment.py --nbr_top_features=15 --output_dir=example_experiment
```

<div align="justify">To generate the results of the best predictive features exploration, run the <code>report_results.py</code> script from the <code>slurm_queue_time_pred.explore_features</code> module using the same arguments as those used to launch the experiment. It is possible to choose a different number of accumulated features to produce the results files. An additional argument, <code>--synthetic_data</code>, if present, allows the use of loss references (MSE) associated with synthetic data. The script will generate a JSON file containing the progression of training, validation and test losses as well as the frequency and average of times these losses deteriorated with the addition of a new feature. It will also generate three graphs representing each of the losses according to the accumulation of the best features.
<br></br>
Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/slurm_queue_time_pred/explore_features/report_results.py --nbr_top_features=10 --output_dir=example_experiment --synthetic_data
```
