library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)


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


## Split data for CV
folds <- vfold_cv(trainCsv, v = 5, repeats=1)

## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a mode

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over
preg_tuning_grid <- grid_regular(penalty(),
                                 mixture(),
                                 levels = 5) ## L^2 total tuning possibilities

## Run the CV
preg_models <- preg_wf %>%
tune_grid(resamples=folds,
          grid=preg_tuning_grid,
          metrics=metric_set(rmse),
          control = untunedModel) # including the control grid in the tuning ensures you can
# call on it later in the stacked model

##Random Forest Model
rf_spec <-rand_forest(mtry= tune(),
                      min_n= tune(), 
                      trees = 500) %>%
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_spec) 



tuning_grid_rf <- grid_regular(mtry(c(25,30)),
                            min_n(),
                            levels = 3) ## L^2 total tuning possibilities

rf_models <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid_rf,
            metrics=metric_set(rmse),
            control = untunedModel) # including the control grid in the tuning ensures you can
# call on it later in the stacked model


## Create other resampling objects with different ML algorithms to include in a stacked model, for example
lin_reg <- linear_reg() %>%
 set_engine("lm")
lin_reg_wf <- workflow() %>%
add_model(lin_reg) %>%
 add_recipe(my_recipe)
lin_reg_model <-fit_resamples(
              lin_reg_wf,
              resamples = folds,
              metrics = metric_set(rmse),
              control = tunedModel)
              


## Specify with models to include
my_stack <- stacks() %>%
 add_candidates(preg_models) %>%
 add_candidates(lin_reg_model) %>%
 add_candidates(rf_models)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset


## If you want to build your own metalearner you'll have to do so manually
## using
stackData <- as_tibble(my_stack)

## Use the stacked data to get a prediction
bike_predictions <- stack_mod %>% predict(new_data=testCsv)
 

cappedBikePred <- mutate(bike_predictions, .pred = ifelse(.pred < 0, 0, .pred))
testCsvBind <- read_csv("test.csv", col_types=c(datetime="character"))
sampleSub1 <- cbind(testCsvBind$datetime,cappedBikePred)
sampleSub1 <- rename(sampleSub1,datetime = 'testCsvBind$datetime', count=.pred) %>%
  mutate(count = exp(count))
             
              
write_csv(sampleSub1, "stackModel.csv")
