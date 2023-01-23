# Data

<div style="text-align: justify">The data is found in the **data** module at the root of the project and is separated into folders according to format and clusters.
<br><br>
Data collection is spread over several consecutive days. Every day, at the same time, a snapshot of the status of all uncompleted jobs on the cluster is captured. This moment of capturing the state of the cluster, or snapshot, is represented by the variable _poll time_.
<br><br>
The variable to be predicted is _poll wait sec_, which corresponds to the difference, in seconds, between the moment when we take the snapshot and the moment when the job actually starts running on the compute cluster. Taking the difference between the current time and the start time of a job on the cluster allows us to measure the time remaining before the job runs. This takes into account the time that has passed since the job was submitted. Therefore, this difference can be considered as equivalent to the waiting time of the job.
<br><br></div>

## Data composition

<div style="text-align: justify">The data consists of csv files, each representing a snapshot of the state of jobs on the Cedar and Graham clusters at 12:00 p.m. (noon) for a given day in the months of May, June and July 2021. In total, 68 files were provided for Cedar, and 61 files for Graham. Cedar's total record count for all days is 1,125,293, while Graham's is 334,748. A record corresponds to the state of a particular job. This state is described by 97 variables, among which 16 have been obtained by the `sacct` command and 81 are "augmented", i.e. they have been calculated from other variables or retrieved a posteriori. The 16 variables obtained by `sacct` characterize the current job, while the others characterize the state of the cluster with respect to the current job.
<br><br>
Some variables are not used for training, either because they come from a linear transformation of poll_wait_sec, therefore to be excluded from the poll_wait_sec prediction task, or because they are redundant (i.e. different units). Variables that are not available at prediction time are removed. In the end, 76 variables are used for training. These variables, which we will call features, are listed in the features.py file of the code.wait_time_prediction module.
<br><br></div>

## Data distribution

<div style="text-align: justify">The plot below shows the distribution of the variable to be predicted, _poll wait sec_, without transformation (top graph) and with a base-10 logarithmic transformation (bottom graph). 
<br><br>
<p align="center">
  <img src="../results/plots/dist_poll_wait_sec.png">
</p>
<br>
We note that without transformation, the distribution of the variable is unbalanced: there is a concentration of points between 0 and 0.25 (values ​​are multiplied by 10<sup>6</sup>), which corresponds to an interval from 0 to 69.4 hours. It is in this interval that the majority of the jobs are found, with a few exceptions represented by the lower bands. 
<br><br>
From this observation came the idea of ​​predicting the order of magnitude of the waiting time, instead of its exact value. This not only improves training by decreasing the prediction error for values ​​far from the majority in the untransformed distribution, but also makes predictions that have greater utility. Indeed, for a user of the computing cluster, it is more relevant to obtain an accurate estimate of the job’s queue time with a coarse granularity (in terms of hours or fractions of an hour) rather than get a rough estimate with fine granularity (in terms of seconds).
<br><br>
We find that with the logarithmic transformation (bottom graph), the distribution approximates more to a normal distribution. There is less concentration around the same values. We chose a base 10 logarithmic transformation for its ease of interpretation, but we could have chosen any other logarithmic transformation without affecting the solution. It would therefore have been possible, for example, to take a natural logarithmic transformation and then add a multiplicative constant to the loss after training the model.
<br><br>
We have also chosen to standardize all the input variables, so that the values ​​are under the same scale. The motivation for this is to have a more stable optimization problem, which is different from our motivation to transform the variable to be predicted. To show this transformation, we chose a feature arbitrarily. The plot below represents the distribution of this feature without transformation and with a standard normalization, so that the mean of the values ​​is 0 and the variance is 1.
<br><br>
<p align="center">
  <img src="../results/plots/dist_eligible_time.png">
</p>
<br></div>