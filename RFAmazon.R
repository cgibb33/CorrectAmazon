library(tidymodels)
library(embed)
library(vroom)
library(lme4)


train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)

forest_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%  
  step_other(all_nominal_predictors(), threshold = .1) %>%  # Label encoding
  step_dummy(all_nominal_predictors())  # Normalize numeric predictors

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model
  set_engine("ranger") %>% # What R function to use
  set_mode("classification")

## Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(rf_mod)

## Set up grid of tuning values
grid_of_tuning_params_forest <- grid_regular(mtry(range = c(1,9)),
                                             min_n(),
                                             levels = 4)


forestfolds <- vfold_cv(train, v = 4, repeats=1)

CV_results_forest <- forest_wf %>%
  tune_grid(resamples=forestfolds,
            grid = grid_of_tuning_params_forest,
            metrics = metric_set(roc_auc)) #Or leave metrics NULL


collect_metrics(CV_results_forest) %>%
  filter(.metric=="roc_auc") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

bestTune_forest <- CV_results_forest %>%
  select_best(metric = "roc_auc")


finalforest_wf <-
  forest_wf %>%
  finalize_workflow(bestTune_forest) %>%
  fit(data=train)


forest_predict <- finalforest_wf %>%
  predict(new_data = test)

rf_submission <- forest_predict %>%
  bind_cols(test) %>%
  rename(ACTION = .pred_class) %>%  # Rename the predicted column to 'ACTION'
  select(id, ACTION)  # Select 'id' and 'ACTION' columns for the submission
  
 

vroom_write(x=rf_submission, file="./ForestAmazon.csv", delim=",")
