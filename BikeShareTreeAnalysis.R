library(tidymodels)
library(tidyverse)
library(vroom)
library(baguette)

set.seed(1234)

#Calls test data
testCsv <- vroom('test.csv')

#Calls training data
trainCsv <- vroom('train.csv') 

##Clean up the weather variables
##Changed the singular value of weather = 4 to weather = 3
trainCsv <- mutate(trainCsv, count = log(count), weather = ifelse(trainCsv$weather == 4, 3,trainCsv$weather)) %>%
  select(-casual,-registered,-temp)

testCsv <- mutate(testCsv, weather = ifelse(weather == 4, 3,weather )) %>%
  select(-temp)



##Feature Engineering (2) recipe()
##Changed variables to factors, added an hour variable, and selected 
##for all variables except casual and registered
my_recipe <- recipe(count ~., data=trainCsv) %>% 
  step_date(datetime,features =c("year")) %>%
  step_time(datetime,features = c("hour"),keep_original_cols = F) %>%
  step_mutate(season=as.factor(season), 
              holiday=as.factor(holiday),
              weather = as.factor(weather),
              year = as.factor(datetime_year),
              hour = as.factor(datetime_hour),
              workingday = as.factor(workingday)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_mutate(int2_6 = hour_X18 * season_X2,
              int3_6 = hour_X18 * season_X3,
              int4_6 = hour_X18 * season_X4,
              ) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

prepped_recipe <- prep(my_recipe)

bake(prepped_recipe, new_data = trainCsv)


#Decision Tree Model
dTree_mod <- decision_tree(tree_depth = tune(),
                           cost_complexity = tune(),
                           min_n = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression")

dTree_wf <-  workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(dTree_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(),
                            cost_complexity(),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(trainCsv, v = 5, repeats=1)


## Run the CV
CV_results <- dTree_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse, mae, rsq)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")
bestTune

## Finalize the Workflow & fit it
final_wf <- dTree_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

bike_predictions <- final_wf %>%
  predict(new_data = testCsv)

#Bagging model

tree_spec <- bag_tree(mode="regression") %>%
  set_engine("rpart", times = 1000) %>%
  set_mode("regression")
tree_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_spec) %>%
  fit(data=trainCsv)

bike_predictions <- predict(tree_wf, new_data=testCsv)

##Random Forest model

rf_spec <-rand_forest(mtry= 30,
                      min_n= 21, 
                      trees = 1000) %>%
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_spec) %>%
  fit(data=trainCsv)

bike_predictions <- predict(rf_wf, new_data=testCsv)

## Grid of values to tune over
tuning_grid <- grid_regular(#mtry(range=c(30,34)),
                            min_n(),
                            levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(trainCsv, v = 5, repeats=1)


## Run the CV
CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(rmse)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()


## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best("rmse")
bestTune

#mtry = 10, min_n = 2
#30 21

## Finalize the Workflow & fit it
final_wf <- rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=trainCsv)

bike_predictions <- final_wf %>%
  predict(new_data = testCsv)

##Caps the predictions at 0, makes datetime a character, and binds the data for submission
cappedBikePred <- mutate(bike_predictions, .pred = ifelse(.pred < 0, 0, .pred))
testCsvBind <- read_csv("test.csv", col_types=c(datetime="character"))
sampleSub1 <- cbind(testCsvBind$datetime,cappedBikePred)
sampleSub1 <- rename(sampleSub1,datetime = 'testCsvBind$datetime', count=.pred) %>%
  mutate(count = exp(count))



#Writes it into a csv file

#write_csv(sampleSub1, "dTreeKaggleSubmission.csv")

#write_csv(sampleSub1, "baggRegKaggleSubmission.csv")

write_csv(sampleSub1, "rfRegKaggleSubmission.csv")


