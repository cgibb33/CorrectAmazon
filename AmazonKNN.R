### KNN Model
knn_recipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())

knn_model <- nearest_neighbor(neighbors= tune()) %>% # set or tune
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(knn_recipe) %>%
add_model(knn_model)

knn_tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

## Split data for CV
knnfolds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
knn_CV_results <- knn_wf %>%
  tune_grid(resamples=knnfolds,
            grid=knn_tuning_grid,
            metrics=metric_set())

## Find Best Tuning Parameters
knn_bestTune <- knn_CV_results %>%
  select_best()

## Finalize the Workflow & fit it
knn_final_wf <-
  knn_wf %>%
  finalize_workflow(knn_bestTune) %>%
  fit(data=train)

predict(knn_wf, new_data=train, type="prob")
