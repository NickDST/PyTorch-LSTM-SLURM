

## Pytorch LSTM

Because this code was run on a SLURM system with a continuous runtime of 4 hours at a time (for GPU),
I programmed it to set up checkpoints and store the information.


The train-test-split process had to be manually programmed due to PyTorch's method might have variation
on each boot-up, so to ensure that the training dataset is separate from the test dataset, the initial split was predetermined from the set seed and the procedural shufflings are determined from an array of set seeds so that once the program
is halted at 4 hours, then it could start up again and pick up right where it started.


The final model uses embedding dimensions of 128 and 2 hidden layers of size 300. 
