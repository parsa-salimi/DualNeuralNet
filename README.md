# DualNeuralNet

This repository implements code to solve combinatorial problems using reinforcement learning
the ES folder modifies code by A Z Wagner : https://github.com/zawagner22/cross-entropy-for-combinatorics
the agents folder uses [tf-agents](https://github.com/tensorflow/agents) and implements REINFORCE and DQN.

 The code works with:
 ```
 python version 3.6
 numpy version 1.19.5
 tensorflow version 2.4.1
 ```
 
 # Compute Canada instructions
 First, we need to load the following modules:
 
 ```
 module load scipy-stack
 module load python/3.6
 ```
 
 Then, create a virtual environment by typing
 
 `virtualenv --no-download [env-name]`
 
 A directory `[env-name]` will be created in the current working directory.  Now activate the virtual environment:
 
 `source [env-name]/bin/activate`
 
 Finally, update pip and install tensorflow:
 
 ```
 pip install --upgrade pip
 pip install tensorflow_gpu
 ```
 
 To run the code in the `agents` folder, install `tf-agents`:
 
 `pip install tf-agents`
 
 We are now ready to run jobs. This is usually done in a batch file as follows:
 
 ```
 #!/bin/bash
#SBATCH --gres=gpu:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=RNN8Freq.out 

module load python/3.6
module load scipy-stack
source [env-name]/bin/activate
module load cuda cudnn
python DualNeuralNet/ES/mainRNN.py  #or mainFNN.py
 ```
