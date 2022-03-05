library(CVXR)
require(ggplot2)

elastic_reg <- function(beta, lambda = 0, alpha = 0) {
  ridge <- (1 - alpha) / 2 * sum(beta^2)
  lasso <- alpha * p_norm(beta, 1)
  lambda * (lasso + ridge)
}



set.seed(1)

# Problem data
p <- 20
n <- 1000
DENSITY <- 0.25    # Fraction of non-zero beta
beta_true <- matrix(rnorm(p), ncol = 1)
idxs <- sample.int(p, size = floor((1 - DENSITY) * p), replace = FALSE)
beta_true[idxs] <- 0

sigma <- 45
X <- matrix(rnorm(n * p, sd = 5), nrow = n, ncol = p)
eps <- matrix(rnorm(n, sd = sigma), ncol = 1)
y <- X %*% beta_true + eps


TRIALS <- 10
beta_vals <- matrix(0, nrow = p, ncol = TRIALS)
lambda_vals <- 10^seq(-2, log10(50), length.out = TRIALS)
beta <- Variable(p)  


loss <- sum((y - X %*% beta)^2) / (2 * n)
obj <- loss + elastic_reg(beta, lambda, alpha)

loss <- huber(y - X %*% beta, M)


loss <- sum((y - X %*% beta)^2) / (2 * n)

## Elastic-net regression
alpha <- 0.75
beta_vals <- sapply(lambda_vals,
                    function (lambda) {
                      obj <- loss + elastic_reg(beta, lambda, alpha)
                      prob <- Problem(Minimize(obj))
                      result <- solve(prob)
                      result$getValue(beta)
                    })