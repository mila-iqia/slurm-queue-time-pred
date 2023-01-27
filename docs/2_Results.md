# Results


## Linear model

<div align="justify">
The following plot shows the predictions obtained after training the linear model according to the true jobs’ queue times (right graph). The black line that cuts the plane in half on the diagonal corresponds to the result that we would obtain if the model made no prediction errors. The red hatched lines are used to position real (non-logarithmic) wait time values on the x-axis.
<br></br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/plots/predictions_plot_all_linear.png">
   </td>
  </tr>
 </table>
 <i>Predictions obtained with the linear model, after training on Cedar data.
 </i>
</div>
<br>
The training, validation and test losses with the linear model, on a base 10 logarithmic scale, are 0.4046, 0.8315 and 0.4224 respectively. We see in the figure above that the model has trouble predicting values that are below 3, which corresponds to about 20 minutes of waiting (10<sup>3</sup> seconds = 20 minutes). Indeed, few examples of jobs with a waiting time (poll_wait_sec) of 20 minutes or less are present in the dataset, which explains this difficulty.
<br></br>
</div>

## Non-linear model


### With data from Cedar

<div align="justify">The following figure shows the predictions obtained from the 10 separate training runs of the neural network trained on Cedar data according to true wait times.
<br></br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/plots/predictions_plot_1000_7NN.png">
   </td>
  </tr>
 </table>
 <i>Wait time predictions obtained with the neural network model after training on Cedar data (1000 randomly chosen examples).
 </i>
</div>
<br>

The training, validation, and test losses are calculated by taking the average of 10 training runs of the model. These are the MSE based on the differences between <code>log10(pred)</code> and <code>log10(target)</code> values.
<br><br>
<div align="center">
 
| | MSE on log10 values |
|-|---------------------|
|train| 0.2139 |
|valid| 0.3673 |
|test|  0.4299 |

</div>
<br></br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/plots/error_distribution_7NN.png">
   </td>
  </tr>
 </table>
 <i>Distribution of differences (in log10) between predictions and actual values of wait time for 10 separate training runs
 of the neural network trained on Cedar data.
 </i>
</div>
<br><br>

If we express that in terms of our original units of time, we have that:

- 38.19% of predicted *wait_time* falls into the interval [0.50*t, 2.0*t] with t being the real wait time in seconds,
- 56.76% of predicted *wait_time* falls into the interval [0.33*t, 3.0*t] with t being the real wait time in seconds.

<br>
Note that we are talking here about differences between predictions and original values without logarithmic transformation. That is to say, for a job whose execution is predicted in 2 hours, there is about a 38% chance that the job will actually be running on the SLURM cluster in an interval of [1 hour, 4 hours].
<br><br>
</div>

### With data from Graham

<div align="justify">The following figure shows the predictions obtained from 10 separate training runs of the neural network trained on Graham data according to the true wait times.
<br><br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/plots/predictions_plot_1000_6NN.png">
   </td>
  </tr>
 </table>
 <i>Wait time predictions obtained with the neural network model after training on Graham data (1000 randomly chosen examples).
 </i>
</div>
<br>
The training, validation, and test losses are calculated by taking the average of 10 training runs of the model. These are the MSE based on the differences between <code>log10(pred)</code> and <code>log10(target)</code> values.
<br><br>
<div align="center">
 
| | MSE on log10 values |
|-|---------------------|
|train| 0.2579 |
|valid| 0.4332 |
|test|  0.6583 |

</div>
<br></br>
The following histogram shows the distribution of differences for the 10 model training runs.
<br><br>
<div align="center">
 <table>
  <tr>
   <td><img src="../results/plots/error_distribution_6NN.png">
   </td>
  </tr>
 </table>
 <i>Distribution of differences (in log10) between predictions and actual values of wait time for 10 training runs of the neural network trained on Graham data.
 </i>
</div>
<br>
Values below -5.0 are not shown for readability. These are 292 predictions below -5.0 that have been omitted from the histogram above out of a total of 150,420. As before, we can calculate the percentage of predictions that fall into the intervals [0.50*t, 2.0*t] and [0.33*t, 3.0*t]. This percentage is 41.60% and 59.89% respectively.
<br><br>
</div>

## Comparison with SLURM estimates

<div align="justify">As a comparison, we can use the difference between the actual execution time of the jobs on the compute clusters and the value predicted by SLURM. This is equivalent to the difference between the actual and predicted waiting time. Indeed, SLURM provides an estimate based, among other things, on the time limit requested by the user for the job and the priority of other users’ jobs submitted afterwards. It is well known, however, that SLURM's estimation is grossly inaccurate.
<br></br>
We have run our own informal experiment on the Cedar and Graham clusters. We can get a estimate for the wait time for a job by using the <code>--test-only</code> flag when submitting a job. This does not run the actual job, however, so if we want to get an actual measurement we need to submit that job again, wait for it to run, and record the time that it took. We have generated a variety of job requirements and computed the differences between the estimate and the reality.
<br></br>
We can report differences using the same MSE loss (on log10 values) as before to have the same basis of comparison. From our n=53 estimates on Cedar we get an MSE of 3.2032, and with n=58 estimates on Narval cluster we get an MSE of 2.4870.
<br></br>
We can do the same exercise as before and also report the intervals where >50% of the predictions fall into, where t is the real wait time in seconds:
<br></br>
<div align="center">

| | Cedar | Narval |
|-|-------|--------|
|Mean difference| 3.2032 | 2.4870 |
|>50% prediction interval| [0.0006t, 1596.61t] | [0.0033*t, 306.9*t] |

</div>
<br></br>
Dummy job data for the Cedar and Narval clusters is located in the project's <code>slurm_queue_time_pred.start_time_estimation</code> module.
</div>
