library(dplyr)
library(bnlearn)
library(gRain)

setwd("Desktop/Bayesian Data Analysis/Final Proj/")

dat <- read.table("caselistingNew.txt", header = T, nrows = 70000, sep = "", na.strings = ".") # These are the fields
# that correspond to the screenshot.

# Getting more descriptive names for dat:
new.names <- c("obs", "state", "case.num", "vehicle.num",
               "person.num", "atm.cond", "num.fatalities", 
               "speeding", "age",
               "airbags", "alcohol.t.r", "alcohol.t.s",
               "alcohol.t.t", "drug.t.r", "drug.t.s", "drug.t.t",
               "injury.severity",
               "meth.alc.det", "meth.drug.det",
               "drug.involvement", "alc.involvement",
               "race",
               "seatbeltORhelmet", "sex",
               "num.fat.vehicle", "num.occupants", "travel.speed", "vehicle.make", "vehicle.model", "driver.alc.involvement",
               "driver.height.ft", "driver.height.in", "driver.weight.lb", "speed.limit")

names(dat) <- new.names

# Replace NAs with -1s:
dat <- as.data.frame(apply(X = dat, MARGIN=2, FUN = function(x) replace(x,which(is.na(x)),-1)))

# Get drivers only, eliminate useless variables
dat <- filter(dat, person.num==1)
data <- select(dat, -obs, -case.num, -vehicle.num, -person.num)
rm(dat)

# Separate data into two dfs to allow for discretize() to work:
apply(X = data, MARGIN=2, FUN = function(x) length(unique(x))) > 3 -> x
data[,x] <- as.data.frame(apply(X = data[,x], MARGIN=2, FUN = function(x) x+runif(n = length(x), min = -0.001, max=0.001)))

# Hartemink method
data[,x] <- discretize(data = data[,x], method = "hartemink", breaks = 3, ibreaks = 30, idisc="quantile")

# Turn others into factors:
data[,-x] <- as.data.frame(apply(X=data[,-x], MARGIN=2, FUN = as.factor))

# Load above
load(project_code.RData)

set.seed(666)
rand.rows <- sample(size = round(0.8*dim(data)[1]), x = (1:dim(data)[1]), replace = F)
train <- data[rand.rows,]
test <- data[-rand.rows,]

# Nodes to use:
used <- c("alcohol.t.r", "alcohol.t.t", "alcohol.t.s", "alc.involvement", 
  "meth.alc.det", "speeding", "seatbeltORhelmet", "airbags", "sex", 
  "injury.severity", "num.fat.vehicle", "num.occupants")

# Reduce workspace:
rm(data, x, new.names, rand.rows)

# Model 1
# Manual
bn.structure <- dag(
  c("alcohol.t.r", "alcohol.t.t", "alcohol.t.s", "alc.involvement"),
  c("alcohol.t.t", "meth.alc.det"),
  c("speeding", "alcohol.t.r"),
  c("seatbeltORhelmet", "airbags", "sex"),
  c("injury.severity", "seatbeltORhelmet"),
  c("num.fat.vehicle", "num.occupants", "speeding", "injury.severity")
)

grain.manual <- grain(x = bn.structure, train, smooth = 1)
grain.manual

# Model 2, structure learning with BIC.
# Learn a model in bnlearn
bnl.mod <- hc(x = train[,used], score = "bic")

bnl.mod.fit <- bn.fit(bnl.mod, train[,used], method = "bayes")

# Model 3
library(deal)
deal.net <- network(train[,used])
prior <- jointprior(deal.net, N=10)
deal.net <- learn(deal.net, train[,used], prior)$nw
deal.best <- autosearch(deal.net, train[,used], prior, removecycles = T)

# Export to bnlearn
bnlearn.deal <- model2network(modelstring(deal.best$nw))
bnlearn.deal.fit <- bn.fit(bnlearn.deal, train[,used], method = "bayes")

# Plots
bnl.manual <- as.bn(grain.manual)
graphviz.plot(bnl.manual, shape = "ellipse")
dev.print(pdf, file="manual.pdf")

graphviz.plot(bnl.mod, shape = "ellipse")
dev.print(pdf, file="learnedBIC.pdf")

graphviz.plot(bnlearn.deal, shape = "ellipse")
dev.print(pdf, file="deal.pdf")

# Test data:
set.seed(666)
newdata <- sample_n(test, size = 8000, replace = F)

# Exact Predictions with the three models, p(injury.severity | sex, airbags, num.fat.vehicle)
# Manual
preds1 <- predict(object = grain.manual, predictors = c("sex", "airbags", "num.fat.vehicle"),
                       response = "injury.severity", newdata = newdata)
sum(unlist(preds1$pred)==as.character(newdata$injury.severity))/dim(newdata)[1]
# 0.705

# Model learned via BIC score
preds2 <- predict(object = as.grain(bnl.mod.fit), predictors = c("sex", "airbags", "num.fat.vehicle"),
                  response = "injury.severity", newdata = newdata)
sum(unlist(preds2$pred)==as.character(newdata$injury.severity))/dim(newdata)[1]
# 0.770

# Fully Bayesian Model
preds3 <- predict(object = as.grain(bnlearn.deal.fit), predictors = c("sex", "airbags", "num.fat.vehicle"),
                  response = "injury.severity", newdata = newdata)
sum(unlist(preds3$pred)==as.character(newdata$injury.severity))/dim(newdata)[1]
# 0.770

# Demonstrating propagation via "sex"
# Model 1
grain.manual.sex <- setFinding(grain.manual, nodes = "sex", states = "(-0.99903,1.99903]")
querygrain(grain.manual.sex)$injury.severity
querygrain(grain.manual)$injury.severity

# Model 2
bnl.mod.fit.sex <- setFinding(as.grain(bnl.mod.fit), nodes = "sex", states = "(-0.99903,1.99903]")
querygrain(bnl.mod.fit.sex)$injury.severity
querygrain(as.grain(bnl.mod.fit))$injury.severity

# Model 3
bnlearn.deal.fit.sex <- setFinding(as.grain(bnl.mod.fit), nodes = "sex", states = "(-0.99903,1.99903]")
querygrain(bnlearn.deal.fit.sex)$injury.severity
querygrain(as.grain(bnlearn.deal.fit))$injury.severity
