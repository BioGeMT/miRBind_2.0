# miRBind 2.0

Repository for ML tools for miRNA binding site prediction.

### Structure of the repository:

- `code` - Folder with plain scripts, that can work on their own and are reusable. Example of such a script is *specific encoder for the miRNA:mRNA sequences*; or a *random Forest training script*.
- `analysis` - Folder with directory per each specific analysis, eg. *training of Random Forest on K mer encoding of Manakov 1:1 dataset*. Each of this specific analysis will contain one master script, that will reproduce the whole analysis that was performed. 
You can also use your analysis folder as a playground (eg. for hyperparameter optimization or trying different architectures), but the master script should run just the final analysis.
Store here intermediate results too, when running the analysis (like encoded datasets or prediction files), but please don't commit them.
All the scripts in this folder should not be code-heavy, they should use code chunks already available in the `code` directory. 
Use custom code pieces only for parts that are very specific fo your analysis and not reusable. 
- `data` - Placeholder empty folder, that locally contains all the datasets used for model trainings etc.
Here on GitHub it contains just a script that when executed will download the datasets and put them in a proper folder structure.
- `models` - Placeholder empty folder, similar to the `data` folder. Contains a script that downloads the trained models.


### How to work:

Create a new branch, work there, and when you are ready, clean it up and create a pull request to the main branch.

##### Scenario 1: adding shared utility
If you want to add a new utility (like data encoder, model architecture, evaluation script) that can be used by others, put it in the `code` folder.
Make it ideally a plain python script, that can be run on its own and is reusable.

##### Scenario 2: creating a new analysis
You might already have all the chunks you need coded and you want to just unite them into a single analysis. Or you might want to experiment a bit.
Then create a folder in the `analysis` folder with a descriptive name of the analysis you are doing. 
You can try there for example different hyperparameter settings or different training dataset ratios. But at the end, you should have a master script that will run the final pipeline with specific settings and produce one consistent result.

#### Scenario 3: changing shared utility / dataset / model
It might happen that you eg. find a bug or want to make faster some of the shared code pieces in the `code` folder. 
Then change the code, but also make sure to find all its usages in the `analysis` folder and update them accordingly / rerun the analysis / put there a flag (maybe an issue) so people know that their results might be outdated.

The same thing applies when there is an update in a **dataset** or to a **model**.

### Models Leaderboard:

Here we track the performance of ML models across Manakov22 test and leftout datasets. Models are ranked based on the sum(AUPRC(test), AUPRC(leftout)).

#### Leaderboard

| Rank | Model                              | AUPRC(test)  | AUPRC(leftout) | link to model  | link to code  |
|------|------------------------------------|--------------|----------------|----------------|---------------|
| 1    | Pairwise encoding CNN              | 84.97        | 83.08          | [model](https://drive.google.com/drive/folders/1dFsm0CcC7WL2mP4h5a6UZtVt57ICH3vB?usp=drive_link) | LINK |
| 2    | retrained miRBind CNN (published in miRBench)  | 84.00        | 81.00          | LINK | LINK |
| 3    | TargetScanCNN                      | 77.00        | 76.00          | LINK | LINK |
| 4    | ...                                | 00.0         | 00.0           | LINK | LINK |
| 5    | ...                                | 00.0         | 00.0           | LINK | LINK |
| 6    | ...                                | 00.0         | 00.0           | LINK | LINK |
| 7    | ...                                | 00.0         | 00.0           | LINK | LINK |
| 8    | ...                                | 00.0         | 00.0           | LINK | LINK |

#### How to Submit Your Model
Follow these steps:
1. Push your code to reproduce and evaluate the model to GitHub
2. Create a new folder for your trained model in [Google Drive](https://drive.google.com/drive/folders/1IH7_CjxWW7Q0dKEFJY3L3yo4B2WWxJh2?usp=drive_link) and upload it
3. Fill in the table with metrics and links to the model and code (the path to the folder containing the code and guidelines to run it is sufficient)
