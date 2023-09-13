##Data Wrangling
##Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)


trainCsv <- vroom('train.csv') 

##Clean up the weather variables
##Changed the singular value of weather = 4 to weather = 3
trainCsv <- mutate(trainCsv, weather = ifelse(trainCsv$weather == 4, 3,trainCsv$weather ))

trainCsv
##Feature Engineering (2) recipe()
##Changed variables to factors, added an hour variable, and selected 
##for all variables except casual and registered
my_recipe <- recipe(count ~., data=trainCsv) %>% 
  step_mutate(season=as.factor(season), 
              holiday=as.factor(holiday),
              weather = as.factor(weather),
              workingday= as.factor(workingday)) %>%
  step_time(datetime,features = c("hour"),keep_original_cols = F) %>%
  step_select(-casual,-registered)
prepped_recipe <- prep(my_recipe)

bake(prepped_recipe, new_data = trainCsv)
