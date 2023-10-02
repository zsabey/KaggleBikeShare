##

library(tidyverse)
library(vroom)
library(patchwork)

sampleSub <- vroom('sampleSubmission.csv')

testCsv <- vroom('test.csv')

trainCsv <- vroom('train.csv')

trainCsv

dplyr::glimpse(trainCsv)

skimr::skim(trainCsv)

view(trainCsv)

trainCsv$season <- as.factor(trainCsv$season)
trainCsv$holiday <- as.factor(trainCsv$holiday)
trainCsv$workingday <- as.factor(trainCsv$workingday)
trainCsv$weather <- as.factor(trainCsv$weather)


DataExplorer::plot_intro(trainCsv)
corrPlot1 <- DataExplorer::plot_correlation(select(trainCsv, count, windspeed, humidity, atemp, temp), theme_config = list(legend.position = "none"))

DataExplorer::plot_bar(trainCsv)
DataExplorer::plot_correlation(data = trainCsv,theme_config = list(legend.position = "none"))
histPlot1 <- DataExplorer::plot_histogram(trainCsv)
?plot_correlation
DataExplorer::plot_missing(trainCsv)
GGally::ggpairs(trainCsv)

tempvCountPlot <- ggplot(data=trainCsv) +
  geom_point(mapping=aes(x=temp,y=log(count))) +
  geom_smooth(mapping=aes(x=temp,y=log(count)), se=FALSE)

tempvCountPlot


weathervCountPlot <- ggplot(data=trainCsv) + 
  geom_boxplot(mapping=aes(x=weather,y=log(count)))
weathervCountPlot

(corrPlot1 + histPlot1) / (tempvCountPlot + weathervCountPlot)

ggplot(data=trainCsv) +
  geom_point(mapping=aes(x=humidity,y=count))            

