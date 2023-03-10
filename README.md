# SLURM WAIT TIME PREDICTION

## Purpose

<div align="justify">The objective of the work is to allow a better orchestration of the jobs on the HPC SLURM system by predicting the waiting time for these jobs, i.e. the time between the moment when the user submits the job and the moment when the job starts to run, in order to improve the use of the computing clusters. Currently, SLURM provides an estimate of job start times, but the accuracy of these estimates is not considered acceptable by the DRAC. 
<br></br>
Deep learning, a machine learning technique, is leveraged to obtain wait time predictions. In particular, multi-layer perceptrons are used. More complex models, such as models with attention, were not used due to the limitations of the project, namely its duration and the nature of the data obtained.
</div>


## Setup

<div align="justify">The code can be executed either from this directory (same path as this README.md), or by setting the <code>PYTHONPATH</code> environment variable.
</div>

```
export PYTHONPATH=$PYTHONPATH:`pwd`
# from anywhere
python3 -m slurm_queue_time_pred.wait_time_prediction.run_experiment 
```
Alternatively,
```
# from inside this repo
python3 slurm_queue_time_pred/wait_time_prediction/run_experiment.py
```


## Main results

<div align="justify">On average, our neural network model trained on data from the Cedar computing cluster makes predictions 
approximately 1 594 times closer to the jobs' actual wait times than SLURM's estimator. More than 50% of our model's predictions
fall in the interval [ 0.37 * target, 2.69 * target ]. By our experimental measurements, when using SLURM's estimator the equivalent interval is [ 0.0006 * target, 1596.61 * target ] which makes SLURM's estimator useless in practice.
<br></br>
The best predictors of a job's wait time on the cluster appear to be the amount of memory, GPUs and CPU cores already in use on the computing cluster by other users' jobs at submission.
</div>

## Documentation

 <font size="4"><a href="docs/0_Data.md">Data</a><br>
<a href="docs/1_Methods.md">Methods</a><br>
<a href="docs/2_Results.md">Results</a><br>
<a href="docs/3_Train_on_past_data.md">Training on Past Data</a><br>
<a href="docs/4_Feature_exploration.md">Feature Exploration</a><br>
<a href="docs/5_Discussion.md">Discussion</a><br>
</font>


## Contributors

<div align="justify">This project was carried out as part of the internship of Bianca Popa, software engineering student at the ??cole de technologie sup??rieure, from August 29, 2022 to January 27, 2023 at Mila, under the guidance of Guillaume Alain, deep learning researcher. Bianca Popa mainly contributed with the implementation, model training and documentation while Guillaume Alain mainly contributed with his ideas and expertise.
<br></br>
The project was done in collaboration with Pier-Luc St-Onge from Calcul Qu??bec and James Desjardins and Tyler Collins from SHARCNET, under the umbrella of the Digital Research Alliance of Canada (DRAC). 
<br></br>
Special thanks to these individuals for collecting and sharing the data on the use of the Cedar and Graham computing clusters, because the project could not have been carried out otherwise. Also, big thanks to Mila and the IDT team for the internship opportunity and for access to computational resources.
</div>

## Citation

```
@software{Popa_SLURM_queue_time_2023,
 author = {Popa, Bianca and Alain, Guillaume},
 month = {1},
 title = {{SLURM queue time prediction using neural networks}},
 version = {1.0.0},
 year = {2023}
}
```
