# Load necessary libraries
library(tidymodels)
library(embed)
library(vroom)
library(lme4)

# Load the datasets
train <- vroom("train.csv")
test <- vroom("test.csv")

# Convert ACTION to a factor (target variable)
train$ACTION <- as.factor(train$ACTION)

# Define the recipe for preprocessing
my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = as.factor) %>% # Ensure numeric predictors are factors if necessary
  step_other(all_nominal_predictors(), threshold = 0.02) # Group low-frequency factor levels

# Define the logistic regression model
logRegModel <- logistic_reg() %>%
  set_engine("glm")

# Create a workflow: combine recipe and model
logreg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel)

# Fit the workflow to the training data
logreg_fit <- fit(logreg_wf, data = train)

# Predict probabilities on the test data
amazon_predictions <- predict(logreg_fit, new_data = test, type = "prob")

# Bind the predictions with the test set and prepare submission
kaggle_submission <- amazon_predictions %>%
  bind_cols(test) %>%
  rename(ACTION = .pred_1) %>%  # Rename the predicted column to 'ACTION'
  select(id, ACTION)  # Select 'id' and 'ACTION' columns for the submission

# Write the submission file
vroom_write(x = kaggle_submission, file = "./amazon_log.csv", delim = ",")



#Penalized Logistic Regression

penlog_recipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

prep <- prep(penlog_recipe)
bake(prep, train)

penlog_mod <- logistic_reg(mixture= 0, penalty=.4) %>% #Type of model
  set_engine("glmnet")

penlog_workflow <- workflow() %>%
add_recipe(penlog_recipe) %>%
add_model(penlog_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- penlog_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
select_best()

## Finalize the Workflow & fit it
final_wf <-
penlog_workflow %>%
finalize_workflow(bestTune) %>%
fit(data=train)

## Predict
penlogfinal <- final_wf %>%
predict(new_data = test, type="prob")

penlog_submission <- penlogfinal %>%
  bind_cols(test) %>%
  rename(ACTION = .pred_1) %>%  # Rename the predicted column to 'ACTION'
  select(id, ACTION)  # Select 'id' and 'ACTION' columns for the submission

# Write the submission file
vroom_write(x = penlog_submission, file = "./amazon_penlog.csv", delim = ",")

