##Data Wrangling
##Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)
library(poissonreg)
library(pscl)
library(glmnet)

#Calls test data
testCsv <- vroom('test.csv')

#Calls training data
trainCsv <- vroom('train.csv') 

##Clean up the weather variables
##Changed the singular value of weather = 4 to weather = 3
trainCsv <- mutate(trainCsv, weather = ifelse(trainCsv$weather == 4, 3,trainCsv$weather)) %>%
  select(-casual,-registered,-temp)

testCsv <- mutate(testCsv, weather = ifelse(weather == 4, 3,weather )) %>%
  select(-temp)


##Feature Engineering (2) recipe()
##Changed variables to factors, added an hour variable, and selected 
##for all variables except casual and registered
my_recipe <- recipe(count ~., data=trainCsv) %>% 
  step_mutate(season=as.factor(season), 
              holiday=as.factor(holiday),
              weather = as.factor(weather),
              workingday= as.factor(workingday)) %>%
  step_time(datetime,features = c("hour"),keep_original_cols = F) %>%
  step_dummy(all_nominal_predictors()) %>% #make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean 0, sd=1

prepped_recipe <- prep(my_recipe)

bake(prepped_recipe, new_data = trainCsv)

##Linear Regression in R

my_mod <- linear_reg() %>%
  set_engine("lm")

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = trainCsv)

bike_predictions <- predict(bike_workflow, 
                            new_data = testCsv) # Use fit to predict

##Poisson Regression in R
pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(pois_mod) %>%
fit(data = trainCsv) # Fit the workflow


# Print the model summary

bike_predictions <- predict(bike_pois_workflow,
                            new_data=testCsv) # Use fit to predict

# Penalized regression model
preg_model <- poisson_reg(penalty=0, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(preg_model) %>%
fit(data=trainCsv)
bike_predictions <- predict(preg_wf, new_data=testCsv)



##Caps the predictions at 0, makes datetime a character, and binds the data for submission
#expBikePred <- mutate(bike_predictions, .pred = exp(.pred))
cappedBikePred <- mutate(bike_predictions, .pred = ifelse(.pred < 0, 0, .pred))
testCsvBind <- read_csv("test.csv", col_types=c(datetime="character"))
sampleSub1 <- cbind(testCsvBind$datetime,cappedBikePred)
sampleSub1 <- rename(sampleSub1,datetime = 'testCsvBind$datetime', count=.pred)


#Writes it into a csv file
#write_csv(sampleSub1, "linearRegKaggleSubmission.csv")

write_csv(sampleSub1, "poissonRegKaggleSubmission.csv")

#write_csv(sampleSub1, "pRegKaggleSubmission.csv")

