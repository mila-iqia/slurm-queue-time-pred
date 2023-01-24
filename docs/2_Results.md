# Results


## Linear model

The losses calculated with the exact solution, on a base 10 logarithmic scale, are 0.3174 for training, 0.7462 for validation and 0.4134 for testing. 
<br><br>
The following figure shows the predictions obtained with the least squares solution according to the true jobs’ queue times (left graph) and the predictions obtained after training the linear model according to the true jobs’ queue times (right graph). The black line that cuts each of the planes in half on the diagonal corresponds to the result that we would obtain if the model made no prediction errors. The red hatched lines are used to position real (non-logarithmic) wait time values on the x-axis.
<br><br>
<p align="center">
 <img src="../results/plots/predictions_plot_all_lstsqr.png" width="100">
 <img src="../results/plots/predictions_plot_all_linear.png" width="100">
</p>
<p align="center">
 <i>Predictions obtained with the least squares solution (left) and after training the linear model (right). All examples from the Cedar dataset are included.
 </i>
</p>
<br>
The training, validation and test losses with the linear model, on a base 10 logarithmic scale, are 0.4046, 0.8315 and 0.4224 respectively. We see in the figure above that the model has trouble predicting values ​​that are below 3, which corresponds to about 20 minutes of waiting (10<sup>3</sup> seconds = 20 minutes). Indeed, few examples of jobs with a waiting time (poll_wait_sec) of 20 minutes or less are present in the dataset, which explains this difficulty. 

Overall, we find that the model trains correctly, since the predictions obtained after training are very close to the predictions obtained with the least squares.


## Non-linear model


### With data from Cedar

	

The following figure shows the predictions obtained from the 10 training trials of the 7-layer model described in section 3.2 according to true wait times.


![drawing](https://docs.google.com/drawings/d/12345/export/png)

<i>Figure 4. Waiting time predictions obtained with the 7-layer model after training on Cedar data (1000 randomly chosen examples).</i>

The training, validation, and test losses calculated by taking the average of 10 training trials of the model, on a base-10 logarithmic scale, are 0.2139, 0.3673, and 0.4299, respectively. To give a better idea of ​​what this represents, we can calculate the percentage of times the prediction error does not exceed a factor of 2 and a factor of 3. In other words, we want the percentage of predictions that fall in the interval [ -2 * target, 2 * target ] and in the interval [ -3 * target, 3 * target ]. This percentage is calculated from the distribution of the differences between the actual (target) values ​​and the predictions. The following histogram shows the distribution of differences for the 10 model training trials.



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


_Figure 5. Distribution of differences (in log10) between predictions and actual values ​​of waiting time for 10 training repetitions of the 7-layer model._

The percentage of predictions lower than a factor of 2 times the targets is 38.19% and the percentage of predictions lower than a factor of 3 times the targets is 56.76%. Note that we are talking here about differences between predictions and original values ​​without logarithmic transformation. That is to say, for a job whose execution is predicted in 2 hours, there is about a 38% chance that the job will actually be running on the Slurm cluster in an interval of [30 minutes, 4 hours].


#### 4.2.2. With data from Graham

	The following figure shows the predictions obtained from 10 training trials of the 6-layer model described in section 3.2 according to the true wait times.



<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)

_Figure 6. Waiting time predictions obtained with the 6-layer model after training on Graham data (1000 randomly chosen examples)._

The training, validation, and test losses calculated by taking the average of 10 training trials of the model, on a base-10 logarithmic scale, are 0.2579, 0.4332, and 0.6583, respectively. The following histogram shows the distribution of differences for the 10 model training trials.



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)

_Figure 7. Distribution of differences (in log10) between predictions and actual values ​​of waiting time for 10 training repetitions of the 6-layer model._

Values ​​below -5.0 are not shown for readability. These are 292 predictions below -5.0 that have been omitted from the histogram above out of a total of 150,420. As before, we can calculate the percentage of predictions below a factor of 2 times the targets and a factor of 3 times the targets. This percentage is 41.60% and 59.89% respectively.


### 4.3. Comparison with SLURM estimates

As a comparison, we can use the difference between the actual execution time of the jobs on the compute clusters and the value predicted by SLURM. This is equivalent to the difference between the actual and predicted waiting time. Indeed, SLURM provides an estimate based, among other things, on the time limit requested by the user for the job and the priority of other users’ jobs submitted afterwards. It is well known, however, that SLURM's estimation is grossly inaccurate.

We determined that the mean difference, on a base-10 logarithmic scale, for n=53 estimates on the Cedar cluster is 3.2032, while for n=58 estimates on the Narval cluster it is 2.4870. To calculate these values, fictitious jobs were submitted on these clusters and for each of them, the estimated job execution start time and the actual execution start time were retrieved. A difference of 3.2032 corresponds to a factor of approximately 1597 times the targets and a difference of 2.4870 corresponds to a factor of approximately 307 times the targets.

Dummy job data for the Cedar and Narval clusters is located in the project's code.start_time_estimation module.