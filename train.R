library(tidyverse)
library(data.table)
library(caret)
library(gridExtra)
library(ggplot2)
library(randomForest)
library(xgboost)
library(lightgbm)
library(corrplot)
library(rpart)
# Load data
data <- fread("Indian_House_Prices.csv")
head(data)

summary(data)

data %>% count(City, sort = TRUE)

# create a copy of data
data_for_corr <- data

# Change to factor int
data_for_corr$City <- as.factor(data_for_corr$City)
data_for_corr$Location <- as.factor(data_for_corr$Location)

# Convert factor to numeric
data_for_corr$City <- as.numeric(data_for_corr$City)
data_for_corr$Location <- as.numeric(data_for_corr$Location)

# Check for missing values
colSums(is.na(data))

# Correlation matrix remove the first column
correlation_matrix <- cor(data_for_corr[, -1])
corrplot(correlation_matrix, method = "number")

# Select useful columns
Data <- data %>%
  select(City, Location, Area, `No. of Bedrooms`, CarParking, AC, Wifi, LiftAvailable, `24X7Security`, Price) %>%
  rename(total_sqft = Area, 
         BHK = `No. of Bedrooms`, 
         Parking = CarParking, 
         Lift = LiftAvailable, 
         Security = `24X7Security`)
head(Data)

data %>% count(City, sort = TRUE)

# Split data into training and testing
set.seed(123)

train_index <- createDataPartition(data$Price, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Changing the data types of some columns to integer
Data <- Data %>%
  mutate(across(c(BHK, Parking, AC, Wifi, Lift, Security), as.integer))

head(Data)

# Checking for null values
Data %>% summarise(across(everything(), ~sum(is.na(.))))

# Outlier Removal
Data <- Data %>%
  filter(total_sqft / BHK >= 300)

# Transforming less frequent areas to 'Other' here less than 10 is threshold
temp <- Data %>% count(Location, sort = TRUE)
temp1 <- filter(temp, n < 10)
Data <- Data %>%
  mutate(Location = if_else(Location %in% temp1$Location, "Other", Location))

head(Data)

ggplot(Data, aes(x = factor(BHK), y = Price, fill = City)) +
  geom_bar(stat = "summary", fun.y = "mean", position = "dodge") +
  theme_minimal() +
  labs(title = "Price for different number of Bedrooms across various cities")

# rename the Location to Area
Data <- Data %>%
  rename(Area = Location)

# Feature Engineering also create a dictionary of the factor for each city and Area
Data <- Data %>%
  mutate(City = as.factor(City), Area = as.factor(Area))

# create a dictionary for storing factor and cites and Area as key-value pairs (used for app.R)
# Key = factor, Value = City/Area
City_dict <- Data %>%
  select(City) %>%
  distinct() %>%
  mutate(City = as.character(City)) %>%
  mutate(City = paste0("", City))

Area_dict <- Data %>%
  select(Area) %>%
  distinct() %>%
  mutate(Area = as.character(Area)) %>%
  mutate(Area = paste0("", Area))

# Save them as csv
write.csv(City_dict, "City_dict.csv", row.names = FALSE)
write.csv(Area_dict, "Area_dict.csv", row.names = FALSE)


# Encoding categorical variables
Data <- Data %>%
  mutate(across(c(City, Area), ~ as.integer(factor(.))))

# Splitting data into training and testing sets
set.seed(0)
trainIndex <- createDataPartition(Data$Price, p = .8, 
                                  list = FALSE, 
                                  times = 1)
DataTrain <- Data[ trainIndex,]
DataTest  <- Data[-trainIndex,]

X_train <- DataTrain %>% select(-Price)
y_train <- DataTrain$Price
X_test <- DataTest %>% select(-Price)
y_test <- DataTest$Price

# Create models dir
dir.create("models")

# Linear Regression
lm_model <- lm(Price ~ ., data = DataTrain)
summary(lm_model)
y_pred <- predict(lm_model, DataTest)
postResample(y_pred, y_test)

# Save rds file
saveRDS(lm_model, "models/lm_model.rds")


# Decision Tree
tree_model <- rpart(Price ~ ., data = DataTrain, method = "anova")
y_pred <- predict(tree_model, DataTest)
summary(tree_model)
postResample(y_pred, y_test)
saveRDS(tree_model, "models/DT_model.rds")

# Random Forest
rf_model <- randomForest(Price ~ ., data = DataTrain, ntree = 50)
y_pred <- predict(rf_model, DataTest)
postResample(y_pred, y_test)
summary(rf_model)
saveRDS(rf_model, "models/RF_model.rds")

# XGBoost
xgb_model <- xgboost(data = as.matrix(X_train), label = y_train, nrounds = 50, objective = "reg:squarederror")
y_pred <- predict(xgb_model, as.matrix(X_test))
postResample(y_pred, y_test)
summary(xgb_model)
saveRDS(xgb_model, "models/XGB_model.rds")

# LightGBM
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
params <- list(objective = "regression", metric = "rmse", learning_rate = 0.1, num_leaves = 31)
lgb_model <- lgb.train(params, lgb_train, 50)
y_pred <- predict(lgb_model, as.matrix(X_test))
postResample(y_pred, y_test)
saveRDS(lgb_model, "models/LGB_model.rds")


# Hyperparameter tuning using GridSearchCV
#train_control <- trainControl(method="cv", number=5)
#tune_grid <- expand.grid(mtry = c(2, 3, 4, 5),
 #                        min.node.size = c(1, 3, 5),
  #                       splitrule = c("variance", "extratrees"))

#tuned_rf <- train(Price ~ ., data = DataTrain,
    #              method = "ranger",
   #               tuneGrid = tune_grid,
     #             trControl = train_control)
# Model Evaluation
#print(tuned_rf)
#y_pred <- predict(tuned_rf, DataTest)
#postResample(y_pred, y_test)
