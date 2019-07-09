This project is part of my master thesis at the Technical University Munich and also part of the MaLSAMi project of the Chair for Operating Systems at the Department of Informatics.

# Recurrent Neural Network (RNN) based Schedulability Analysis
Schedulability analysis (SA) using a recurrent neural network (RNN). 



# Data
The Task-Sets are given through a SQL-database with the following three tables:

- TaskSet: Set_ID, Successful, TASK1_ID, TASK2_ID, TASK3_ID, TASK4_ID
- Task: Task_ID, Priority, Deadline, Quota, CAPS, PKG, Arg, CORES, COREOFFSET, CRITICALTIME, Period, Number_of_Jobs, OFFSET
- Job: Set_ID, Task_ID, Job_ID, Start_Date, End_Date, Exit_Value

# Installation and Start
Download or clone the hole project. Add the database as described above to the project directory. Change to the project directory and type  
```bash
vagrant up
```
to start the vagrant machine. Then type
```bash
vagrant ssh
``` 
to ssh into the vagrant machine. Change the directory with
```bash
cd /vagrant
``` 
to the project directory. Then you can do different things:
- do hyperparameter exploration
- train and evaluate a single model
- plot experiment results

# Hyperparameter Exploration
For hyperparameter exploration uncomment line 66 in [main.py](./main.py) and specifiy a name and 
number for the experiment (also name of the resulting csv-file):
```python
hyperparameter_exploration(data=data, name='This is the name of the experiment', num='This is the
 number of the experiment')
```
The hyperparameter, that are tested, are defined in [params.py](./params.py). Currently the 
following hyperparameters are included:

Hyperparameter | Description
--- | ---
batch_size | Size of the data batches (number of task-sets), for which the weights are updated
num_epochs | Number of epochs, in which the hole dataset is processed
keep_prob | Percentage of inputs that are forwarded; (1 - keep_prob) are the number of inputs that are not forwarded by the dropout layer
num_cells | number of cells (LSTM, GRU) or recurrent layers
hidden_layer_size | size of the layers (number of neurons per layer)
hidden_activation | activation function of the layers
optimizer | algorithm to optimize the weights

There are also some configuration parameters defined in this file, to specifiy the hyperparameter
 experiment:
 
Configuration Parameter | Description
--- | ---
use_checkpoint | if the ModelCheckpoint callback should be used (saves the model)
checkpoint_dir | directory where the model should be saved
checkpoint_verbose | how much information should be printed to the console during saving of the model
use_earlystopping | if the EarlyStopping callback should be used (stops training if no improvement)
use_tensorboard | if the TensorBoard callback should be used (collects information for TensorBoard)
tensorboard_log_dir | directory where the TensorBoard log-files should be saved
use_reduceLR | if the RecudeLROnPlateau callback should be used (adapts learning rate automatically)
verbose_training | how much infomration should be printed to the console during training
verbose_eval | how much information should be printed to the console during evaluation
time_steps | number of time steps = sequence length = maximum number of tasks per task-set
element_size | sequence vector length = number of attributes per task
num_classes | number of classes = number of bits for coding the classes

The hyperparameter exploration can then be started by typing
```bash
python3.6 main.py
```
in the console. The results of the experiment can be found in the csv-file created in the working
 directory.
 
 # Train and Evaluate a Single Model
 To train and evaluate a single Keras model uncomment line 72 in [main.py](./main.py) and start 
 the program by typing
 ```bash
python3.6 main.py
```
in the console. The hyperparameters and configuration parameters can be specified in the file 
[params.py](./params.py).

# Plot Experiment Results
The plotting functions are defined in the file [plotting.py](.\plotting.py). Add the 
desired functions to the main()-function and start the programm by typing
```bash
python3.6 plotting.py
```
in the console.
Currently there are the following plotting functions available:
- plot a single hyperparameter (single line plot)
- plot the confusion matrix
- plot the correlation matrix