**How to download all necessary libraries:**

We have made a text file that contains all relevant libraries. Run "pip install -r requirments.txt" in the terminal to install the packages listed.
For mac users: It might be necessary to install libomp since it is a depedency for xgboost. If you do not already have it you can use the command "brew install libomp".

**How to run the code:**

You need to create an access token for the tabpfn_client library. You do this by going to https://priorlabs.ai and clicking try TabPFN now, then you sign in/login. On your site you have your personal access token under API-keys. Create a .env file on your pc and write TABPFN_TOKEN= you personal token.

We have provided all the datasets in the datasets/ folder, and all forest diffusion models in the folder forest_diffusion_models/.

If you do not want to regenerate the datasets and the models, simply run model_selection.ipynb. If you want to run xgboost_imputation.ipynb without generating new diffusion models, you need to switch block 19 to use the commented-out load_models function and not the forest_diffusion_training dunction. To understand the data it is ideal to read preprocessing.ipynb

If you do want regenerate everthing you need to delete:
- all datasets in datasets/ except train.csv and test.csv. These are the raw datasets provided by kaggle.
- forest_diffusion_models folder
  
And then run the notebooks in this order:
1. preprocessing.ipynb
2. xgboost_imputation.ipynb
3. model_selection.ipynb


**Competition Description**

We are partaking in a kaggle contest named Spaceship Titanic. Our task is to predict whether a passenger was transported to another dimension during the spacship's collision. We are given a training and a test dataset, so that is our starting point. We are aiming to find a good model that gets a great score on the unseen test dataset. The test dataset we are given does not contain a label so we need to create a csv file with passengerid and the predictions for the test dataset and send it to kaggle to receive our testscore.


**Novel methods**

Thale - TabPFN : TabPFN is a pretrained transformer, trained on synthetic data and optimized for small to medium sized tabular datasets. I implemented tabpfn using the tabpfn_client library, following the standard workflow from https://github.com/PriorLabs/tabpfn-client (under basic usage).  Three different models were trained on three different datasets(raw, preprocessed, augmented) using .fit(), and the validation accuracy was evaluated using accuracy_score where the arguments were true y_val and the result from model.predict(X_val) for each of the datasets. The results were stored in a dict alongside all the other models results for comparison. The TabPFN model trained on the preprocessed dataset achieved the best validation score overall.

Aurora - RandomRotationEnsemble: RandomRotationEnsemble is a Wrapper used to apply the random rotations of the feature space to each of the base learners in en ensemble of decision trees. Depending on the "base_learner" parameter it creates a randomly rotated Random Forest, a randomly rotated Extra Trees or lastly an ensemble of decision trees where their only difference is the inherent randomness within the model and their different rotations. It is implemented using numpy, pandas, sklearn and scipy. The rotations are only applied to the numerical features, which are found by taking all features with at least 10 unique values. Given that the splitting of data by bootstrapping is built into the RandomForest model from sklearn, we opted to use DecisionTreeClassifires as base learners and adding the boostrapping step ourselves to create the effect og Bagging. The rotation matrix for each base learner is stored, as when making predictions on new data each base learner has to receive the new data rotated with the according matrix.

Eirik - Forest Diffusion description of functions:
forest_diffusion_training() , which takes care of the main pipeline of the forest-flow method. This involves duplicating the dataset 100 times and generating an equal DataFrame of random noise for each of the duplicates. This is to ensure a diverse representation of data/noise for every data point. Then, these are joined in n different ratios of (1/n) to (1-(1/n)) and a dedicated model is trained for each noise level. The task of these models is to predict the trajectory leading to a noise level  smaller than the current one. 

The forward functions are used to "corrupt" clean data by progressively mixing it with Gaussian noise over time t, which is used to generate noisy training targets for the XGBoost models at each level. 

The reverse function inverts the process by using trained models to predict and subtract noise at each step to denoise a sample back toward a clean data point. 

The forest_diffusion_repaint_imputation() function takes a clean dataset with missing values, generates a sample of random noise, and for each denoising model, noises the clean data to the appropriate level, inserts the generated samples into the missing value slots and performs a denoising step. Then, original data is restored and the process is repeated until we pass through the model corresponding to the least amount of noise. At this point, the dataset consists of the original one, with generated samples derived from random noise filling missing values.

This allows us to train classifiers as we can remove missing values without loosing data, but the XGBoost diffusion models can also be used to generate completely synthetic data, which might improve generalization. This is done by generating a complete dataset full of noise and passing it through one model after the other, from most noisy to least. One limitation of data generation is that it is not able to replicate hard domain rules like people in CryoSleep not being able to spend money. To deal with this, postprocessing is applied to the relevant column and remaining data is rescaled to normal values with respects to the original dataset.

Lastly, the augmented dataset comprising equal parts imputed original data and synthetic samples (1:1 ratio, 13908 total samples) is processed to be in the same format as the baseline manually imputed non-augmented version for evaluation.
