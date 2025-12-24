```r
library(readr)
library(dplyr)
library(ggplot2)
library(lmtest)
library(sandwich)
library(tseries)
library(vars)

df <- read_csv("data.csv") # adjust
df$Date <- as.Date(df$Date)

# Scatter + linear fit
ggplot(df, aes(x=Marketing_Spend, y=Revenue)) +
  geom_point(alpha=0.5) + geom_smooth(method="lm")

# Correlations
cor.test(df$Marketing_Spend, df$Revenue, method="pearson")
cor.test(df$Marketing_Spend, df$Revenue, method="spearman")

# OLS with robust SE
model <- lm(Revenue ~ Marketing_Spend, data=df)
coeftest(model, vcov = vcovHC(model, type = "HC3"))

# Granger causality (requires ts / VAR)
# Convert to ts and use vars::VAR then causality()
```
