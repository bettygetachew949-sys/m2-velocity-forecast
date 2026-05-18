# m2-velocity-forecast
Applied econometrics, machine learning, and macroeconomic data analysis projects using Python, R, and Stata.
---
title: "Betelhem Kebede - FINAL PROJECT"
output:
  word_document: default
  html_document: default
  pdf_document: default
date: "2025-11-25"
---

```{r}
library(dplyr)
library(purrr)
library(tidyverse)
library(stringr)
library(skimr)
library(MASS)
library(class)
library(caret)
library(glmnet)
```


```{r}
cpi <- read.csv("Consumer Price Index for All Urban Consumers.csv")
umcsent <- read.csv("Consumer Sentiment (UMCSENT).csv")
unrate <- read.csv("UNRATE.csv")
vm2 <- read.csv("Velocity of M2 Money Stock (M2V).csv")
real_ir <- read.csv("realir.csv")
indprod <- read.csv("INDPRO.csv")
```

```{r}
my_data_list <- list(
  cpi         = cpi,
  unrate      = unrate,
  real_ir     = real_ir,
  indprod     = indprod,
  vm2         = vm2
)
```

# Some data cleaning and merging
```{r}
merged_data <- reduce(my_data_list, full_join, by = "observation_date")
colnames(merged_data) <- merged_data %>% colnames() %>% str_to_lower()
merged_data$observation_date <- as.Date(merged_data$observation_date)
merged_data <- merged_data %>% drop_na()
```


```{r}
merged_data %>% colnames()
```

# Summary Statistics Table

```{r}
library(psych)
library(dplyr)
library(janitor)


library(flextable)
library(officer)
library(dplyr)
library(psych)


summary_stats <- merged_data %>%
  dplyr::select(-observation_date) %>%
  psych::describe() %>%
  as.data.frame()


ft <- flextable(summary_stats)
ft <- autofit(ft)


doc <- read_docx()

doc <- body_add_par(doc, "Summary Statistics Table", style = "heading 1")
doc <- body_add_flextable(doc, ft)


print(doc, target = "summary_statistics.docx")
summary_stats
```
# Time-Series Plots of Variables

```{r}
library(ggplot2)
library(dplyr)
library(tidyr)

# ---- 1. LONG FORMAT ----
plot_data <- merged_data %>%
  pivot_longer(
    cols = -observation_date,
    names_to = "variable",
    values_to = "value"
  )

# ---- 2. BEAUTIFUL FACETED TIME-SERIES PLOT ----
ggplot(plot_data, aes(x = observation_date, y = value)) +
  geom_line(linewidth = 0.7, alpha = 0.9) +
  facet_wrap(~ variable, scales = "free_y", ncol = 3) +
  labs(
    title = "Macroeconomic and Financial Variables Over Time (1982–2024)",
    subtitle = "Quarterly data from FRED",
    x = "Date",
    y = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12, color = "gray30"),
    strip.text = element_text(face = "bold", size = 13),
    panel.spacing = unit(1, "lines"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_line(color = "gray85"),
    panel.grid.major.y = element_line(color = "gray90")
  )
ggsave("velocity_plots.jpg", width = 12, height = 10, dpi = 300)
```
```{r}
library(corrplot)

cor_matrix <- merged_data %>%
  dplyr::select(reaintratrearat1ye, cpiaucsl, unrate, m2v,indpro) %>%
  cor(use = "pairwise.complete.obs")

corrplot(
  cor_matrix,
  method = "color",
  type = "upper",
  tl.cex = 1.0,
  tl.col = "black",
  addCoef.col = "black",
  number.cex = 0.8,
  col = colorRampPalette(c("#6BAED6", "white", "#DE2D26"))(200),  # nicer blue → white → red
  mar = c(0,0,2,0)
)
title(" ", line = 0.5, cex.main = 1.4)

ggsave("corrplot1.jpg", width = 8, height = 6, dpi = 300)
```

# create classification target - velocity up/down or high regime
```{r}
library(dplyr)

merged_data <- merged_data %>%
  mutate(
    m2v_lag1 = lag(m2v, 1),                         # lag of M2V
    d_cpi = cpiaucsl - lag(cpiaucsl, 1),            # differenced CPI
    d_indpro = indpro - lag(indpro, 1),             # differenced industrial production
    m2v_change = m2v - lag(m2v,1),                  # for direction classification first difference
    m2v_dir = ifelse(m2v_change > 0, "Up", "Down"), # direction variable
    m2v_high = ifelse(m2v >= quantile(m2v,0.75,na.rm=TRUE),"High","Low"), # regime
  ) %>% 
  drop_na()  # removes first row with lag NAs
```

```{r}
merged_data <- merged_data %>%
  mutate(
    m2v_change_lag1 = lag(m2v_change, 1)
  ) %>%
  drop_na()
```

# ADF Stationarity Tests
```{r}
library(tseries)
library(urca)
adf.test(merged_data$m2v, k = 4)
adf.test(merged_data$m2v_change)
adf.test(merged_data$cpiaucsl, k = 4)
adf.test(merged_data$indpro, k = 4)
adf.test(merged_data$reaintratrearat1ye, k = 4)
adf.test(merged_data$unrate, k = 4)
adf.test(merged_data$m2v_lag1, k = 4)
adf.test(merged_data$m2v_change_lag1, k = 4)
adf.test(merged_data$d_cpi)
adf.test(merged_data$d_indpro)
```

### Linear Regression BASELINE OLS MODEL

```{r}
# Linear regression predicting quarter-to-quarter changes in M2 velocity
lm_m2v <- lm(m2v_change ~ reaintratrearat1ye + unrate + d_cpi + d_indpro + m2v_change_lag1,
             data = merged_data)

# Summarize the results
summary_lm <- summary(lm_m2v)


coef_table <- data.frame(
  Variable   = rownames(summary_lm$coefficients),
  Coefficient = summary_lm$coefficients[, "Estimate"],
  Std_Error   = summary_lm$coefficients[, "Std. Error"],
  t_value     = summary_lm$coefficients[, "t value"],
  p_value     = summary_lm$coefficients[, "Pr(>|t|)"]
)

coef_table
```
# Check for multicollinearity using VIF

```{r}
library(car)  
vif(lm_m2v)
```

# LOGISTIC REGRESSION MODEL
```{r}
# Load library
library(caret)
library(dplyr)

# 1. Create lagged M2V variable
merged_data <- merged_data %>%
  arrange(observation_date) %>%                         
  mutate(m2v_lag1 = dplyr::lag(m2v, 1))    

# 2. Difference CPI & IndPro to detrend
merged_data <- merged_data %>%
  mutate(
    d_cpi    = cpiaucsl - lag(cpiaucsl, 1),
    d_indpro = indpro   - lag(indpro, 1)
  )

# Remove first obs lost due to lag/diff
merged_data <- na.omit(merged_data)

# Ensure binary outcome variables are factors
merged_data$m2v_dir  <- as.factor(merged_data$m2v_dir)
merged_data$m2v_high <- as.factor(merged_data$m2v_high)


# Train-test split
set.seed(123)
train_index <- createDataPartition(merged_data$m2v_dir, p = 0.80, list = FALSE)

train_data <- merged_data[train_index, ]
test_data  <- merged_data[-train_index, ]


# Logistic Regression Model WITH lag + differenced variables
logit_dir <- glm(
  m2v_dir ~ m2v_change_lag1 + d_cpi + d_indpro + reaintratrearat1ye + unrate,
  data = train_data,
  family = binomial
)

summary_logit <- summary(logit_dir)
summary_logit
```

```{r}
names(merged_data)
```


```{r}
# Predictions on test set
logit_dir_pred <- predict(logit_dir, newdata = test_data, type = "response")
# Convert to class using 0.5 cutoff
logit_dir_class <- ifelse(logit_dir_pred > 0.5, "Up", "Down")
confusionMatrix(as.factor(logit_dir_class), test_data$m2v_dir)
```
# Logistic Regression Regime Model (High Velocity)

```{r}
logit_regime <- glm(
  m2v_high ~ m2v_change_lag1 + d_cpi + d_indpro + reaintratrearat1ye + unrate,
  data = train_data,
  family = binomial
)

# Summary
summary(logit_regime)

# Predictions
logit_regime_pred <- predict(logit_regime, newdata = test_data, type = "response")


# Convert prediction to factor with proper labels
logit_regime_class <- factor(ifelse(logit_regime_pred > 0.5, "High", "Low"),
                             levels = levels(test_data$m2v_high))  

confusionMatrix(logit_regime_class, test_data$m2v_high)

```

### KNN MODEL 

```{r}
library(caret)
library(class)

# Ensure targets are factors
merged_data$m2v_dir  <- as.factor(merged_data$m2v_dir)
merged_data$m2v_high <- as.factor(merged_data$m2v_high)

# Predictor set (including lag/diff variables)
predictors <- c("m2v_change_lag1", "d_cpi", "d_indpro", "reaintratrearat1ye", "unrate")

# Train/test split (80/20)
set.seed(123)
train_index <- createDataPartition(merged_data$m2v_dir, p = 0.8, list = FALSE)

train_data <- merged_data[train_index, ]
test_data  <- merged_data[-train_index, ]


# Scale predictors 
preProc <- preProcess(train_data[, predictors], method = c("center", "scale"))
train_scaled <- predict(preProc, train_data[, predictors])
test_scaled  <- predict(preProc, test_data[, predictors])


# Direction Model (Up/Down)
k_value <- round(sqrt(nrow(train_data)))  
knn_dir_pred <- knn(train = train_scaled,
                    test = test_scaled,
                    cl = train_data$m2v_dir,
                    k = k_value)

# Confusion matrix for Direction
confusionMatrix(knn_dir_pred, test_data$m2v_dir)


# Regime Model (High Velocity)
knn_regime_pred <- knn(train = train_scaled,
                       test = test_scaled,
                       cl = train_data$m2v_high,
                       k = k_value)

# Confusion matrix for Regime
confusionMatrix(knn_regime_pred, test_data$m2v_high)
```

### Regularized Logistic Regression for Direction and Regime Models with Cross-Validation

```{r}
predictors <- c("reaintratrearat1ye","d_cpi","d_indpro","unrate","m2v_lag1")

# Train-test split (80/20)
set.seed(123)
train_index <- createDataPartition(merged_data$m2v_dir, p=0.8, list=FALSE)
train_data <- merged_data[train_index, ]
test_data  <- merged_data[-train_index, ]


# 3. Cross Validation 
set.seed(123)
tune_grid <- expand.grid(
  alpha  = seq(0,1,by=0.1),
  lambda = 10^seq(-4,1,length=50)
)

# Generate reproducible seed list for resamples
seeds <- vector("list", 10*3 + 1)
for(i in 1:(10*3)) seeds[[i]] <- sample.int(1000, nrow(tune_grid))
seeds[[10*3+1]] <- sample.int(1000, 1)

cv_control <- trainControl(
  method="repeatedcv",
  number=10,
  repeats=3,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  seeds = seeds
)


# Direction (Up/Down)

set.seed(123)
logit_dir_cv <- train(
  x=train_data[,predictors],
  y=train_data$m2v_dir,
  method="glmnet",
  family="binomial",
  metric="ROC",
  tuneGrid=tune_grid,
  trControl=cv_control
)

cat("\n===== Best Params: Direction Model =====\n")
logit_dir_cv$bestTune

pred_dir <- predict(logit_dir_cv, test_data[,predictors])
cat("\n===== Confusion Matrix: Direction (Up/Down) =====\n")
confusionMatrix(pred_dir, test_data$m2v_dir)

# Regime (High/Low)
# ------------------------------------------------------
set.seed(123)
logit_regime_cv <- train(
  x=train_data[,predictors],
  y=train_data$m2v_high,
  method="glmnet",
  family="binomial",
  metric="ROC",
  tuneGrid=tune_grid,
  trControl=cv_control
)

cat("\n===== Best Params: Regime Model =====\n")
print(logit_regime_cv$bestTune)

pred_regime <- predict(logit_regime_cv, test_data[,predictors])
cat("\n===== Confusion Matrix: Regime (High/Low) =====\n")
print(confusionMatrix(pred_regime, test_data$m2v_high))
```


### Coefficients from Direction Logistic Regression
```{r}
best_lambda <- logit_dir_cv$bestTune$lambda
best_lambda
best_alpha <- logit_dir_cv$bestTune$alpha
best_alpha

coef_dir <- coef(logit_dir_cv$finalModel, s = best_lambda)
coef_dir

```

### Coefficients from Regime Logistic Regression
```{r}
best_lambda <- logit_regime_cv$bestTune$lambda
best_lambda
best_alpha <- logit_regime_cv$bestTune$alpha
best_alpha
coef_regime <- coef(logit_regime_cv$finalModel, s = best_lambda)
coef_regime
```


### PCA + Regularized Logistic Regression

```{r}

set.seed(123)
train_index <- createDataPartition(merged_data$m2v_dir, p=0.8, list=FALSE)
train_data <- merged_data[train_index, ]
test_data  <- merged_data[-train_index, ]


# PCA Preprocessing (Fixed Components)

preProc <- preProcess(train_data[, predictors],
                      method=c("center","scale","pca"),
                      pcaComp=3)                 # fixed number of PCs

train_pca <- predict(preProc, train_data[, predictors])
test_pca  <- predict(preProc, test_data[, predictors])


# Setting Reproducible Cross-Validation Seeds
set.seed(123)
tune_grid <- expand.grid(
  alpha  = seq(0,1,by=0.1),
  lambda = 10^seq(-4,1,length=50)
)

# generate stable seed list for repeatedcv
seeds <- vector("list", 10*3 + 1)
for(i in 1:(10*3)) seeds[[i]] <- sample.int(1000, nrow(tune_grid))
seeds[[10*3+1]] <- sample.int(1000,1)

cv_control <- trainControl(
  method="repeatedcv",
  number=10,
  repeats=3,
  classProbs=TRUE,
  summaryFunction=twoClassSummary,
  seeds=seeds
)

# Direction Model (Up/Down)

logit_dir_pca <- train(
  x=train_pca,
  y=train_data$m2v_dir,
  method="glmnet",
  family="binomial",
  metric="ROC",
  tuneGrid=tune_grid,
  trControl=cv_control
)

cat("\n===== PCA Direction Model Best Params =====\n")
print(logit_dir_pca$bestTune)

pred_dir_class <- predict(logit_dir_pca, newdata=test_pca)
cat("\n===== PCA Confusion Matrix (Direction) =====\n")
print(confusionMatrix(pred_dir_class, test_data$m2v_dir))


# Regime Model (High/Low)
set.seed(123)
logit_regime_pca <- train(
  x=train_pca,
  y=train_data$m2v_high,
  method="glmnet",
  family="binomial",
  metric="ROC",
  tuneGrid=tune_grid,
  trControl=cv_control
)

cat("\n===== PCA Regime Model Best Params =====\n")
print(logit_regime_pca$bestTune)

pred_regime_class <- predict(logit_regime_pca, newdata=test_pca)
cat("\n===== PCA Confusion Matrix (Regime) =====\n")
print(confusionMatrix(pred_regime_class, test_data$m2v_high))
```


### KNN with PCA Components
```{r}
library(class)
k_value <- round(sqrt(nrow(train_data))) 


#  Direction Model
knn_dir_pred <- knn(train = train_pca,
                    test = test_pca,
                    cl = train_data$m2v_dir,
                    k = k_value)
confusionMatrix(knn_dir_pred, test_data$m2v_dir)


# Regime Model
knn_regime_pred <- knn(train = train_pca,
                       test = test_pca,
                       cl = train_data$m2v_high,
                       k = k_value)
confusionMatrix(knn_regime_pred, test_data$m2v_high)
```
