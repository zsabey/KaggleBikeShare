##Data Wrangling
##Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(patchwork)


testCsv <- vroom('test.csv')


trainCsv <- vroom('train.csv') 

##Clean up the weather variables
##Changed the singular value of weather = 4 to weather = 3
trainCsv <- mutate(trainCsv, weather = ifelse(trainCsv$weather == 4, 3,trainCsv$weather )) %>%
  select(-casual,-registered)

testCsv <- mutate(testCsv, weather = ifelse(weather == 4, 3,weather ))


##Feature Engineering (2) recipe()
##Changed variables to factors, added an hour variable, and selected 
##for all variables except casual and registered
my_recipe <- recipe(count ~., data=trainCsv) %>% 
  step_mutate(season=as.factor(season), 
              holiday=as.factor(holiday),
              weather = as.factor(weather),
              workingday= as.factor(workingday)) %>%
  step_time(datetime,features = c("hour"),keep_original_cols = F)
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

cappedBikePred <- mutate(bike_predictions, .pred = ifelse(.pred < 0, 0, .pred))
testCsvBind <- read_csv("test.csv", col_types=c(datetime="character"))
head(testCsvBind)
sampleSub1 <- cbind(testCsvBind$datetime,cappedBikePred)
sampleSub1 <- rename(sampleSub1,datetime = 'testCsvBind$datetime', count=.pred)

head(sampleSub1)

write_csv(sampleSub1, "linearRegKaggleSubmission.csv")
