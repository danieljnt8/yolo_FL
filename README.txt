Welcome to the gpu server of KOM.

This server comprises:

- 4x Geforce RTX2080 with 11GB 
- Intel(R) Xeon(R) Silver 4112 CPU @ 2.60GHz
- 92GB RAM
- 7TB HDD

To use this resources you have to submit jobs into the installed SLURM (a workload scheduling system).

Programs that you run WITHOUT a SLURM job:
    - cannot have access to the GPUs
    - can only use a small fraction of the available CPUs
    - can only use a small fraction of the available RAM

Getting started with SLURM might take a short while but is not that hard. All it really takes is some basic Linux cmdline knowledge and a couple of special commands.

The basic idea is that you submit jobs to the scheduler and the scheduler executes them based on their priority.
For that, you have to create a script that specifies which resources you need (e.g. 1 GPU 1CPU 2GB of RAM), it SLURM terminology this is called a Job. 
There is an example job file in your home folder under ~/slurm_example/run_lstm.job.
To submit such a Job to the scheduler you run “sbatch <your_jobfile>”. This queues your job to being executed as soon as the requested resources are available

To check if your script runs you can use “squeue” which shows you all currently submitted jobs. If the state is “R” your job is running. 
Otherwise it either waits for other jobs to complete or is requesting a resource configuration that is not available at the server (e.g. 5  GPUs  or  1TB of RAM). 
The jobs standard output is written into a text file inside your working directory.

If you want to cancel a job use “scancel” either with the job id (see “squeue”) or if you want to cancel all jobs of your user with “scancel -u <your username>”.

For more information regarding slurm please see the corresponding man pages as well as:

    - https://slurm.schedmd.com/documentation.html
     
To check if your process is actually using the GPU:

    - run the command: gpu_job <JOBID>    
    - check ~/gpu_usage.log or /var/log/gpu_usage.log (updated every 15s)
    - check if the process alocated to the gpu is yours (e.g. by matching PIDs with the top command)

To manage python dependencies please use the Conda and if you need to install something please create yourself a custom environment.

For more information see:

    - https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


If you have any questions, please do not hesitate to contact your supervisor.

