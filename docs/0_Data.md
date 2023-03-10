# Data

<div align="justify">The data is found in the <code>data</code> module at the root of the project and is separated into folders according to format and clusters.
<br></br>
Data collection is spread over several consecutive days. Every day at noon, a snapshot of the status of all uncompleted jobs on the cluster is captured. The timestamp at which we capture the state of the cluster, or snapshot, is represented by the feature <i>poll_time</i>.
<br></br>
The variable that we have to predict is <i>poll_wait_sec</i>, which corresponds to the difference, in seconds, between the moment when we take the snapshot and the moment when the job actually starts running on the compute cluster. This quantity cannot be determined yet at the time the snapshot is taken because we need to look into the future in order to determine when a given job will indeed start running. This requires some care in the collection of data, and it also creates two kinds of features: those that can be measured at the time that a job is submitted, and those that can only be known later. Naturally, we want to use the former to make predictions about the latter.
<br></br>
</div>

## Data composition

<div align="justify">The data consists of csv files, each representing a snapshot of the state of jobs on the Cedar and Graham clusters at noon for a given day in the months of May, June and July 2021. In total, 68 files were provided for Cedar, and 61 files for Graham. Cedar's total record count for all days is 1,125,293, while Graham's is 334,748. A record corresponds to the state of a particular job. This state is described by 97 features, among which 16 have been obtained by the <code>sacct</code> command and 81 are "augmented", i.e. they have been calculated from other features or retrieved a posteriori. The 16 features obtained by <code>sacct</code> characterize the current job, while the others characterize the state of the cluster with respect to the current job.
<br></br>
Some features are not used for training, either because they come from a linear transformation of <i>poll_wait_sec</i>, therefore to be excluded from the wait time prediction task, or because they are redundant (i.e. different units). Features that are not available at prediction time are removed. In the end, 76 features are used for training. These features are listed in the <code>features.py</code> file of the <code>slurm_queue_time_pred.wait_time_prediction</code> module.
<br></br>
The data has been anonymized by converting the user identity to an integer, as well as removing many of the identifying features that would otherwise be reported by SLURM. We wanted to keep the ability to determine if two jobs belonged to the same user by comparing the user id because that could potentially be a very useful signal to make predictions, and because this is something that would be available in a real system in production.
<br></br>
</div>


## Data distribution

<div align="justify">The plot below shows the distribution of the output variable, <i>poll_wait_sec</i>, without transformation (top graph) and with a base-10 logarithmic transformation (bottom graph).
<br><br>
<div align="center">
  <table>
  <tr>
    <td><img src="../results/plots/dist_poll_wait_sec.png">
    </td>
  </tr>
  </table>
</div>
<p align="center">
 <i>Distribution of poll_wait_sec (to be predicted). Top, without transformation (1e6 scale). Below, with log10 transformation.
 </i>
</p>
<br>
Without this transformation, the distribution of the variable appears unbalanced. There is a concentration of points between 0 and 0.25 (values ??????are multiplied by 10<sup>6</sup>), which corresponds to an interval from 0 to 69.4 hours. The majority of jobs are found in this interval. It is also intuitively reasonable to expect that wait times would be distributed more uniformly in terms of orders of magnitude (e.g. so many will run in 30 minutes, in an hour, in 2 hours, in 4 hours, and so on). The computational demands of jobs running on large-scale compute cluster are probably also spread more uniformly across orders of magnitudes.
<br><br>
From this observation came the idea of <i>predicting the order of magnitude of the waiting time, instead of its exact value</i>. This has the immediate benefit of stabilizing training by SGD, but we feel that it also makes more useful predictions. Another way to look at predicting the order of magnitude is that, if the mean square error (MSE) was used on target values in the range [0, 1e6], there would be an incentive for conservative guesses to be close to the midpoint of that interval. If 97% of jobs run immediately but 3% of jobs spend a whole month queued, we do not want to have the conservative guess that "a given job will wait a full day". We would rather make a prediction that jobs will run immediately. Using logs and minimizing the MSE on logs better achieves that.
<br><br>
We find that with the logarithmic transformation (bottom graph), the distribution approximates more to a normal distribution. We chose log10 transformation for its ease of interpretation, but we could have chosen any other base without affecting the optimal solution. The MSE would have simply been scaled by a constant factor.
<br><br>
We have also chosen to standardize all the features, so that the values are under the same scale. The motivation for this is to have a more stable optimization problem, which is different from our motivation to transform the output variable. To show this transformation, we represent below the distribution of <i>eligible_time</i>. This plot represents the distribution of this feature without transformation and with a standard normalization, so that the mean of the values is 0 and the variance is 1.
<br><br>
<div align="center">
  <table>
  <tr>
    <td><img src="../results/plots/dist_eligible_time.png">
    </td>
  </tr>
  </table>
</div>
<p align="center">
 <i>Distribution of eligible_time (i.e. time when a job becomes eligible to run). Top, without transformation (1e9 scale). Bottom, with standard normalization.
 </i>
</p>
</div>
