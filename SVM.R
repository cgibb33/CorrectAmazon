library(tidymodels)
library(embed)
library(vroom)
library(lme4)
library(kernlab)

train <- vroom("train.csv")
test <- vroom("test.csv")
train$ACTION <- as.factor(train$ACTION)

SVM_recipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())%>%
  step_pca(all_predictors(), threshold=0.8)

prep <- prep(SVM_recipe)
bake(prep, train)

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kernlab")

SVM_workflow <- workflow() %>%
  add_recipe(SVM_recipe) %>%
  add_model(svmRadial)

## Grid of values to tune over
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- SVM_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best()

## Finalize the Workflow & fit it
final_wf <-
  SVM_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
SVMfinal <- final_wf %>%
  predict(new_data = test, type="prob")

SVM_submission <- SVMfinal %>%
  bind_cols(test) %>%
  rename(ACTION = .pred_1) %>%  # Rename the predicted column to 'ACTION'
  select(id, ACTION)  # Select 'id' and 'ACTION' columns for the submission

# Write the submission file
vroom_write(x = SVM_submission, file = "./amazon_SVM.csv", delim = ",")

