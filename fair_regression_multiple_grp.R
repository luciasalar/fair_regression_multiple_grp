
library(l1tf)
library(CVXR)
require(ggplot2)

sp_data <- data.frame(x = sp500$date,
                      y = sp500$log,
                      l1_50 = l1tf(sp500$log, lambda = 50),
                      l1_100 = l1tf(sp500$log, lambda = 100))


## lambda = 50
y <- sp500$log
lambda_1 <- 50 
beta <- Variable(length(y))
objective_1 <- Minimize(0.5 * p_norm(y - beta) +
                          lambda_1 * p_norm(diff(x = beta, differences = 2), 1))
p1 <- Problem(objective_1)
betaHat_50 <- solve(p1)$getValue(beta)

## lambda = 100
lambda_2 <- 100
objective_2 <- Minimize(0.5 * p_norm(y - beta) +
                          lambda_2 * p_norm(diff(x = beta, differences = 2), 1))

p2 <- Problem(objective_2)
betaHat_100 <- solve(p2)$getValue(beta)


ggplot(data = sp_data) +
  geom_line(mapping = aes(x = x, y = y), color = 'grey50') +
  labs(x = "Date", y = "SP500 log-price") +
  geom_line(mapping = aes(x = x, y = l1_50), color = 'red', size = 1) +
  geom_line(mapping = aes(x = x, y = l1_100), color = 'blue', size = 1)

