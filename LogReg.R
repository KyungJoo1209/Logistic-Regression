# Read in the data
df <- read.table("titanic_project.csv", header=T, sep=",")
head(df)

# split data, 900 obs for train, the rest for test
i <- sample(1:nrow(df),0.861*nrow(df), replace=FALSE)
train <- df[i,]
test <- df[-i,]

x <- Sys.time()
# train a logistic regression model on all the data
glm0 <- glm(survived~pclass,data=df,family='binomial')
y <- Sys.time()

# print the coefficients of the model
glm0$coefficients


# test on the test data
probs <- predict(glm0, newdata=test, type="response")
pred <- ifelse(probs>0.5,'1','0')
acc1 <- mean(pred==as.integer(test$survived))

# accuracy, sensitivity, and specificity
cm1 <- table(test$survived,pred)
confusionMatrix(cm1)


# graph 1
par(mfrow=c(2,2))
plot(glm0)

# graph 2
plot(test$age~test$survived)

# data exploration 1
names(df)

# data exploration 2
head(df)

# data exploration 3
summary(df)

# data exploration 4
summary(glm0)

cat("Accuracy: 0.6507 \nSensitivity: 0.6239 \nSpecificity: 0.7297\n")

time <- y-x
time