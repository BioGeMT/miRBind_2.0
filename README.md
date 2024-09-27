# miRBind 2.0

Repository for ML tools for miRNA binding site prediction.

### Structure of the repository:

Basic division of the model workflows based on the data input.
- `WP1_sequence/`: Work Package 1 is using just a sequence of the miRNA and the target mRNA
- `WP2_conservation/`: Work Package 2 is using the sequences together with a conservation score of the mRNA

The Work Packages than have the following structure:

- `WP1_sequence/`:
    - `encode/`: Contains the scripts for encoding the sequences into inner presentation and readme with description of the encoding technique
    - `train/model_name/`: Contains the scripts for training the model, definition of the model architecture and readme about the model. Replace `model_name` with the name of the model, each model has it's own folder.
    - `evaluate/`: Contains the scripts for evaluating the model, producing scores for each miRNA:target pair
    - `workflow/model_name`: Contains a bash script / snakemake / something else for running the whole workflow for each model, from encoding the sequences to evaluating the model. Also includes readme with short explanation

### How to work:

Create your own branch and do there whatever you want. Typically, you will be probably training a new model with some specific architecture, but you might also just add a new encoder.

When you feel ready, review what you did and be sure to have the following things ready: 

You might have used encoder that is already in the main branch or you might have created a new one.
If you created a new one, it should take as an input a .tsv dataset file and output the encoded dataset (eg. in `.npy` format).

For the model training, create a separate folder in the `train` folder with a clean python file that takes the encoded dataset as an input and outputs the trained model. If you feel like, you can also put the model architecture into a separate file and import it in the training script. Create also a readme with a brief description of the model.

For each model created, there should be a compatible evaluation script that takes the trained model and the test dataset as an input and output the scores for each miRNA:target pair.

Finally, create a workflow script that takes the train and test dataset as an input and runs the whole workflow from encoding the sequences to evaluating the model. Create a separate folder in the `workflow` folder with the script and a readme.

When you are done with the cleanup, create a pull request to the main branch and let others review your code.