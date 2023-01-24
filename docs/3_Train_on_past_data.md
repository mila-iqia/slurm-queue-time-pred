# Training on Past Data

<div style="text-align: justify">Rather than randomly choosing test days from the dataset, another approach is to train the model on all days prior to the test day. In the normal operating environment, only past data is available for training. The model should make predictions about jobs that have not yet been executed. This chronological approach gives us performances that are more akin to those that we could observe in the normal environment.
<br></br>
A new parameter <code>test_day</code> selects the day used to test the model. The days preceding the test day will be part of either the validation set or the training set, in a ratio of 20% and 80% respectively. The designation of either set is randomized to ensure that the distribution of the training and validation day data is similar.
<br></br>
</div>

## Results

<div style="text-align: justify">This section presents the results obtained for each of the test days between the 21st and the last day of data from the Cedar cluster. Rather than testing the model on the last day only, we chose a range of test days to account for potential variability in jobs on the last day compared to previous days. The range starts on the 21st day to have enough data for training. 
<br><br>
The following figure shows the average MSE (y-axis) with and without the standard deviation over the training and test set for a sample of experiments where the model was trained on past data. We used the 7-layer model described in <a href="docs/1_Methods.md"> Methods</a>. The x-axis represents each of the days selected for testing between the 21st and the last day of the dataset. 
<br><br>
<p align="center">
 <table>
  <tr>
   <td><img src="../results/previous_days_results/plots/cedar_average_MSE.png">
   </td>
   <td><img src="../results/previous_days_results/plots/cedar_average_MSE_with_std.png">
   </td>
  </tr>
 </table>
</p>
<p align="center">
 <i>Average training (green curve) and test (yellow curve) MSE on predictions from the 7-layer model when trained with the days before T = test day, with (right) and without (left) the standard deviation, using data from Cedar.
 </i>
</p>
<br>
The following figure shows the median MSE (y-axis) with the minimum and maximum values ​​as well as with the values ​​just above and below the median over the training and test set for the same sample of training experiments.
<br><br>
<p align="center">
 <table>
  <tr>
   <td><img src="../results/previous_days_results/plots/cedar_median_MSE_min_max.png">
   </td>
   <td><img src="../results/previous_days_results/plots/cedar_mid_range_MSE.png">
   </td>
  </tr>
 </table>
</p>
<p align="center">
 <i>Median training (curve in green) and test (curve in yellow) MSE on predictions from the 7-layer model when trained with the days before T = day of test, with the minimum ​​and maximum values (right) and values ​​just below and just above the median (left), using data from Cedar.
 </i>
</p>
<br>
In the figures above, the results for the 66th day of testing have been omitted, since the mean, 16.9086, and median, 8.2979, squared error values ​​over the test set were too distant from the rest. Their presence made the graphs difficult to read. These outliers could be attributed to differences in job characteristics present on that day compared to those on other days of data.
<br><br>
Using the maximum days for training the model (T68), we obtain an average training, validation and test loss of 0.3727, 0.4468 and 0.5956 respectively. The percentage of test predictions below a factor of 2 times the targets and a factor of 3 times the targets is 33.30% and 50.38% respectively. We can also notice that adding days for training the model has no significant impact on the prediction error, which follows a constant trend.
<br><br>
</div>

## Code Documentation

<div style="text-align: justify">To run a training experiment on past data, run the <b>run_experiment.py</b> script from the <b>code.train_on_previous_days</b> module specifying the desired training (hyper)parameters, as listed in <a href="docs/1_Methods.md"> Methods</a>. The features argument does not apply. Here are the additional arguments:
<br></br>
</div>
<table>
 <tr>
  <td>-T, --test_day
  </td>	 	
  <td>Limit between the days corresponding to the data in the training and validation sets and the test day
  </td>
  <td>Default: 3*
  </td>
 </tr>
  <tr>
  <td>--single_run
  </td>	 	
  <td>If present, a single experiment with the specified parameters is executed (optional)
  </td>
  <td>Default: NN*
  </td>
 </tr>
</table>

<div style="text-align: justify">* Corresponds to the minimum value allowed for the test day. One day will thus be allocated for training the model and a second for validation.
<br><br>
Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/code/train_on_previous_days/parallel_run.py --test_day=37 --nbr_layers=3 --single_run
```

<div style="text-align: justify">To generate the results of the training on past days experiments, run the <b>report_results.py</b> script from the <b>code.train_on_previous_days</b> module specifying the (hyper)parameters of the experiments for which the results will be generated in the results.previous_days_results directory, as listed in <a href="docs/1_Methods.md"> Methods</a>. This will generate five graphs with, depending on the test day, the average GPU usage and the average duration of experiments, the average training and test MSE, the average training and test MSE with standard deviation, the median training and test MSE with minimum and maximum values ​​and finally the median training and test MSE with values ​​just above and just below.
<br><br>
Here is an example of running the script from outside the project root:
</div>

```
python3 slurm-queue-time-pred/code/train_on_previous_days/report_results.py --nbr_layers=3 --hidden_size=64 --cluster= graham
```