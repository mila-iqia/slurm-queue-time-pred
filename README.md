# SLURM WAIT TIME PREDICTION

## Purpose

<div align="justify">The objective of the work is to allow a better orchestration of the jobs on the HPC SLURM system by predicting the waiting time for these jobs, i.e. the time between the moment when the user submits the job and the moment when the job starts to run, in order to improve the use of the computing clusters. Currently, SLURM provides an estimate of job start times, but the accuracy of these estimates is not considered acceptable by the DRAC. 
<br></br>
Deep learning, a machine learning technique, is leveraged to obtain wait time predictions. In particular, multi-layer perceptrons are used. More complex models, such as models with attention, were not used due to the limitations of the project, namely its duration and the nature of the data obtained.
</div>


## Setup

<div align="justify">Execute following command at root of project (inside slurm-queue-time-pred) or do your own thing here if you want to run from elsewhere:
</div>

```
export PYTHONPATH=$PYTHONPATH:`pwd`
```
Then:
```
# from inside this repo
python3 slurm_queue_time_pred/wait_time_prediction/run_experiment.py
# from anywhere
python3 -m slurm_queue_time_pred.wait_time_prediction.run_experiment 
```


## Main results

<div align="justify">On average, our neural network model trained on data from the Cedar computing cluster makes predictions 
approximately 1 594 times closer to the jobs' actual wait times than SLURM's estimator. More than 50% of our model's predictions
fall in the interval [ -2.69 * target, 2.69 * target ]. By our experimental measurements, when using SLURM's estimator the equivalent interval is [ -1596.61 * target, 1596.61 * target ] which makes SLURM's estimator useless in practice.
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

<div align="justify">This project was carried out as part of the internship of Bianca Popa, software engineering student at the École de technologie supérieure, from August 29, 2022 to January 27, 2023 at Mila, under the guidance of Guillaume Alain, deep learning researcher. Bianca Popa mainly contributed with the implementation, model training and documentation while Guillaume Alain mainly contributed with his ideas and expertise.
<br></br>
The project was done in collaboration with James Desjardins and Tyler Collins from SHARCNET and Carl Lemaire, Pier-Luc St-Onge and Lucas Nogueira from Calcul Québec, under the umbrella of the Digital Research Alliance of Canada (DRAC). 
<br></br>
Special thanks to these individuals for collecting and sharing the data on the use of the Cedar and Graham computing clusters, because the project could not have been carried out otherwise. Also, big thanks to Mila and the IDT team for the internship opportunity and for access to computational resources.
</div>
