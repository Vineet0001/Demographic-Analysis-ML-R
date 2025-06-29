# Load required libraries 
library(ggplot2) 
library(caTools) 
library(rpart) 
library(rpart.plot) 
library(e1071) 
# Load the dataset 
data("HairEyeColor") 
df <- as.data.frame(HairEyeColor) 
# Convert categorical data to factors 
df$Hair <- as.factor(df$Hair) 
df$Eye <- as.factor(df$Eye) 
df$Sex <- as.factor(df$Sex) 
# Split data 
set.seed(123) 
split <- sample.split(df$Eye, SplitRatio = 0.7) 
train <- subset(df, split == TRUE) 
test <- subset(df, split == FALSE) 
# Decision Tree 
decision_tree_model <- rpart(Eye ~ Hair + Sex, data = train, method = "class") 
predictions_dt <- predict(decision_tree_model, test, type = "class") 
accuracy_dt <- mean(predictions_dt == test$Eye) 
print(paste("Decision Tree Accuracy:", accuracy_dt)) 
# Plot Decision Tree 
rpart.plot(decision_tree_model, main = "Decision Tree for Eye Color Classification") 
# Visualize Decision Tree Predictions 
test$Predicted_Eye_DT <- predictions_dt 
ggplot(test, aes(x = Hair, y = Sex, color = Predicted_Eye_DT, shape = Eye)) + 
geom_jitter(width = 0.2, height = 0.2, size = 3) + 
labs(title = "Decision Tree Classification Results on Eye Color", 
x = "Hair Color", y = "Sex") + 
scale_color_manual(values = c("brown", "blue", "green", "red")) + 
theme_minimal() + 
labs(color = "Predicted Eye Color (DT)", shape = "Actual Eye Color") 
# Linear Regression predicting Frequency 
linear_model <- lm(Freq ~ Hair + Eye + Sex, data = train) 
predictions_lr <- predict(linear_model, test) 
# RMSE for Linear Regression 
rmse_lr <- sqrt(mean((predictions_lr - test$Freq)^2)) 
print(paste("Linear Regression RMSE:", rmse_lr)) 
# Plot Linear Regression predictions vs. actual frequency 
ggplot(test, aes(x = Freq, y = predictions_lr)) + 
geom_point(color = "blue") + 
geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
labs(title = "Linear Regression: Predicted vs. Actual Frequency", 
x = "Actual Frequency", y = "Predicted Frequency") + 
theme_minimal() 
# Polynomial Regression (2nd degree polynomial) 
poly_model <- lm(Freq ~ poly(as.numeric(Hair), 2) + poly(as.numeric(Eye), 2) + Sex, data = train) 
predictions_pr <- predict(poly_model, test) 
# RMSE for Polynomial Regression 
rmse_pr <- sqrt(mean((predictions_pr - test$Freq)^2)) 
print(paste("Polynomial Regression RMSE:", rmse_pr)) 
# Plot Polynomial Regression predictions vs. actual frequency 
ggplot(test, aes(x = Freq, y = predictions_pr)) + 
geom_point(color = "purple") + 
geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") + 
labs(title = "Polynomial Regression: Predicted vs. Actual Frequency", 
x = "Actual Frequency", y = "Predicted Frequency") + 
theme_minimal() 
# Support Vector Machine (SVM) 
svm_model <- svm(Eye ~ Hair + Sex, data = train, kernel = "linear") 
predictions_svm <- predict(svm_model, test) 
accuracy_svm <- mean(predictions_svm == test$Eye) 
print(paste("SVM Accuracy:", accuracy_svm)) 
# K-Means Clustering 
kmeans_model <- kmeans(as.numeric(df$Hair), centers = 3) 
df$Cluster <- as.factor(kmeans_model$cluster) 
# Visualize Clustering Results 
ggplot(df, aes(x = Hair, y = Sex, color = Cluster)) + 
geom_point(size = 3) + 
labs(title = "K-means Clustering Results", x = "Hair Color", y = "Sex") + 
scale_color_manual(values = c("blue", "green", "red")) + 
theme_minimal() + 
labs(color = "Cluster") 
# Output results 
print(paste("Decision Tree Accuracy:", accuracy_dt)) 
print(paste("Linear Regression RMSE:", rmse_lr)) 
print(paste("Polynomial Regression RMSE:", rmse_pr)) 
print(paste("SVM Accuracy:", accuracy_svm)) 
print("Cluster Centers:") 
print(kmeans_model$centers) 



