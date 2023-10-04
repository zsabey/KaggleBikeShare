library(tidymodels)
library(tidyverse)
library(vroom)
library(doParallel)


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
  step_time(datetime,features = c("hour"),keep_original_cols = F) %>%
  step_mutate(season=as.factor(season), 
              holiday=as.factor(holiday),
              weather = as.factor(weather),
              hour = as.factor(datetime_hour),
              workingday = as.factor(workingday)) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

prepped_recipe <- prep(my_recipe)

bake(prepped_recipe, new_data = trainCsv)



##Random Forest model

rf_spec <-rand_forest(mtry= 30,
                      min_n= tune(), 
                      trees = 500) %>%
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_spec) #%>%

# Set the number of cores to use
num_cores <- 4  # Adjust this number based on your system

# Register parallel backend
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Replace the existing code for folds and CV_results with parallel processing

## Grid of values to tune over
tuning_grid <- grid_regular(#mtry(range=c(30,34)),
  min_n(),
  levels = 10) ## L^2 total tuning possibilities

# Create a parallel backend
folds <- vfold_cv(trainCsv, v = 5, repeats = 1)

# Define a function for cross-validation
cv_function <- function(fold) {
  cv_results <- rf_wf %>%
    tune_grid(resamples = list(train = fold),
              grid = tuning_grid,
              metrics = metric_set(rmse, mae, rsq)) %>%
    collect_metrics()
  return(cv_results)
}

# Perform cross-validation in parallel
CV_results <- foreach(fold = folds) %dopar% {
  cv_function(fold)
}

# Stop the parallel backend
stopCluster(cl)

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



write_csv(sampleSub1, "rfRegKaggleSubmission.csv")