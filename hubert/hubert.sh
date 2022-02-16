# ==============================================================================================================================================
# =============================================================== QSUB OPTIONS =================================================================
# ==============================================================================================================================================

# _____ for cpu _____
# act_mem_free - specifies the real amount of free memory (at the time of scheduling).
#                You can specify it when submitting a job and it will be scheduled to a machine with at least this amount of memory free
# mem_free -  specifies a consumable resource tracked by SGE and it affects job scheduling.
#             Each machine has an initial value assigned (slightly lower than the real total physical RAM capacity).
#             When you specify qsub -l mem_free=4G, SGE finds a machine with mem_free >= 4GB, and subtracts 4GB from it.
#             This limit is not enforced, so if a job exceeds this limit, it is not automatically killed and thus the SGE value of mem_free may not represent the real free memory
# h_vmem - hard limit on the size of virtual memory.
#          If your job exceeds this limit, memory allocation fails (i.e., malloc or mmap will return NULL), and your job will probably crash on SIGSEGV.

# _____ for gpu ____
# specify RAM requirements with e.g. -l mem_free=8G,act_mem_free=8G,h_data=12G.
# *Note that you need to use h_data instead of h_vmem for GPU jobs. CUDA driver allocates a lot of “unused” virtual memory (tens of GB per card)*
# gpu=n, n is the number of GPUs
# gpu_ram=XG, where X is GPU mem requirement
# If you need more than one GPU card (on a single machine), always require at least as many CPU cores (-pe smp X) as many GPU cards you need.


#$ -l gpu=1,gpu_ram=13G,mem_free=12G,act_mem_free=12G,h_vmem=16G
# ............................................................................................


# number of CPUs
# If you need more than one GPU card (on a single machine), always require at least as many CPU cores (-pe smp X) as many GPU cards you need.

#$ -pe smp 8
# ............................................................................................


# job name

#$ -N hubert
# ............................................................................................


# working directory

#$ -wd /lnet/express/work/people/stankov/alignment/Thesis
# ............................................................................................


# stdout and stderr outputs are merged and redirected to a file (''script.sh.o$JOB_ID'')

#$ -j y
# ............................................................................................


# priority of your job as a number between -1024 and -100, -100 is default
# priority between -99 and 0 is for urgent jobs taking less than a few hours

#$ -p -200
# ............................................................................................


# Specify the emails where you want to be notified,  -m n to override the defaults and send no emails

#$ -m n
# ............................................................................................


# specify the queue
#  -q 'gpu*'  - any gpu queue
#  -q 'gpu-m*' -  GPU cluster gpu-ms.q at Malá Strana
#  -q '*@hector[14]' to submit on hector1 or hector4,

# machine	      GPU type	    	    cc	      GPU cnt	     GPU RAM (GB)	    machine RAM (GB)
# ------------------------------------------------------------------------------------------
# dll1	     Quadro RTX 5000		    7.5         	8	          16	             366.2
# dll3	     GeForce GTX 1080 	    6.1         	10          11	             248.8
# dll4	     GeForce GTX 1080 	    6.1         	10          11	             248.8
# dll5	     GeForce GTX 1080 	    6.1         	10          11	             248.8
# dll6	     NVIDIA RTX A4000		    8.6         	8	          16	             248.8
# dll7	     GeForce RTX 2080 	    7.5         	8	          11	             248.8
# dll8	     Quadro RTX 5000		    7.5         	8	          16	             366.2
# dll9	     GeForce RTX 3090		    8.6         	4	          25	             183.0
# dll10	     GeForce RTX 3090		    8.6         	4	          25	             183.0

#$ -q 'gpu-ms.q@dll*'


# ================================================================================================================================================
# =============================================================== SHELL COMMANDS =================================================================
# ================================================================================================================================================

# env for environment vars
#env
if [[ "$HOSTNAME" == "dll6.ufal.hide.ms.mff.cuni.cz"  || "$HOSTNAME" == "dll9.ufal.hide.ms.mff.cuni.cz" || "$HOSTNAME" == "dll10.ufal.hide.ms.mff.cuni.cz" ]];
then
  source /lnet/express/work/people/stankov/python-venvs/torch-gpu-3090/bin/activate
  echo ">>>> ampere"
else
  source /lnet/express/work/people/stankov/python-venvs/torch-gpu/bin/activate
  echo ">>>> other"
fi

echo ">>>> running python"
python /lnet/express/work/people/stankov/alignment/Thesis/hubert/huber_model.py
echo ">>>> done"