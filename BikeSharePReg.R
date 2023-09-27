library(tidymodels)
library(vroom)
library(poissonreg) #if you want to do penalized, poisson regression
library(tidyverse)

set.seed(1234)
#Calls test data
testCsv <- vroom('test.csv')

#Calls training data
trainCsv <- vroom('train.csv') 

##Clean up the weather variables
##Changed the singular value of weather = 4 to weather = 3
trainCsv <- mutate(trainCsv, count = log(count), weather = ifelse(trainCsv$weather == 4, 3,trainCsv$weather)) %>%
  select(-casual,-registered)

testCsv <- mutate(testCsv, weather = ifelse(weather == 4, 3,weather )) #%>%
#select(-temp)




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

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model)

## Grid of values to tune over14
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(trainCsv, v = 5, repeats=1)


## Run the CV
CV_results <- preg_wf %>%
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

## Finalize the Workflow & fit it1
final_wf <- preg_wf %>%
finalize_workflow(bestTune) %>%
fit(data=trainCsv)

## Predict
bike_predictions <- final_wf %>%
  predict(new_data = testCsv)


##Caps the predictions at 0, makes datetime a character, and binds the data for submission
cappedBikePred <- mutate(bike_predictions, .pred = ifelse(.pred < 0, 0, .pred))
testCsvBind <- read_csv("test.csv", col_types=c(datetime="character"))
sampleSub1 <- cbind(testCsvBind$datetime,cappedBikePred)
sampleSub1 <- rename(sampleSub1,datetime = 'testCsvBind$datetime', count=.pred) %>%
  mutate(count = exp(count))


#Writes it into new csv

write_csv(sampleSub1, "pRegKaggleSubmission.csv")
