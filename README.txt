This is the code and data setup from our paper "LIR-Net: Learnable Iterative Reconstruction Network for Fan Beam CT Sparse-View Reconstruction".
Readers should note the following information:
1. Iteration_.py contains the basic framework for our training. If you have difficulty using that framework, we recommend that you refer to that file and our paper to build a new training framework. That is easy.
2. "script_radon_identify.py" and "script_radon_learn_inv.py" are the training files for the learnable forward and backward projection operators.
3. The file placement scheme of the datasets is in ". /datasets" and ". /OperatorData" file sequences.
4.". /2020_mayo_patientsID" sequence holds our training set, validation set and test set divisions.
In addition, we store the trained operator weights in ". /OperationResults_" sequence. You can read our operators directly to simulate the projection data. You can refer to the work in "dataTrans.py" (which contains work with operators) for the specific operator reading scheme. If you still have difficulties in reproducing the arithmetic, we suggest you to refer to the paper "AAPM DL-Sparse-View CT Challenge Submission Report: Designing an Iterative Network for Fanbeam-CT with Unknown Geometry". It can help you understand the operator better.
Finally, many thanks to Martin Genzel et al for their previous work on the operator. Part of our work on the operator was referred to the "AAPM DL-Sparse-View CT Challenge Submission Report: Designing an Iterative Network for Fanbeam-CT with Unknown Geometry" for the design.
