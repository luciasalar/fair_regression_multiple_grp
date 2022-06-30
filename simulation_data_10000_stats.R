library(dplyr)
library(tidyr)
library(faux)
library(ltm)
library(fastDummies)
library(Hmisc)
library(corrr)
library(broom)

setClass(Class="Data_Statistics",
         representation(
           outcome ="numeric",
           outcome_model='lm',
           grp_outcome_mean = 'numeric',
           grp_LR_cases = 'numeric',
           systolic_blood_pressure_mean = 'numeric',
           creatinine_mean = 'numeric',
           hematocrit_mean = 'numeric',
           age_mean ='numeric',
           systolic_blood_pressure_sd = 'numeric',
           creatinine_sd = 'numeric',
           hematocrit_sd = 'numeric',
           age_sd ='numeric'
         )
)


generate_data_4groups <- function(number_of_cases, seed){
  set.seed(seed)

    number_of_cases = 10000
    #categorical variables
    sex <- sample(factor(c('male','female')), number_of_cases, replace=TRUE, prob=c(0.5, 0.5))
    race <- sample(factor(c('Black','Hispanic','White','Asian', 'Indigenous')), number_of_cases, replace=TRUE, prob=c(0.29, .25, .23, .18, .05))
    
    #setting errors for different groups, larger error, larger variance in outcome
    error <- ifelse(race == 'White', rnorm(number_of_cases, 0, 1), rnorm(number_of_cases, 0, 1))
    error <- ifelse(race == 'Black', rnorm(number_of_cases, 0, 2), error)
    error <- ifelse(race == 'Hispanic', rnorm(number_of_cases, 0, 1.5), error)
    error <- ifelse(race == 'Indigenous', rnorm(number_of_cases, 0, 1.5), error)
    
    insurance <- sample(factor(c('not_insured', 'private_insured', 'public_insured')), number_of_cases, replace=TRUE, prob=c(.10, .30, .60))
    comorbidities <- sample(factor(c('None', 'one', 'two', 'three_above')), replace=TRUE, number_of_cases, prob=c(.17, .41, .23, .19))
    
    categorical <- data.frame(error, sex, race, insurance, comorbidities)
    
    categorical_dummy <- dummy_cols(categorical, select_columns = c('sex','race','insurance','comorbidities'), remove_selected_columns = TRUE, remove_first_dummy = TRUE)
    
    #creatinine (DR criteria) should be correlated with race, 60 mL - 300 ml 
    Black <- categorical_dummy$race_Black
    
    creatinine_error <- rnorm(number_of_cases, 0, 20)
    creatinine = 300 + 25*categorical_dummy$race_Black + 20*categorical_dummy$race_Hispanic + 20*categorical_dummy$race_Indigenous + 2* categorical_dummy$comorbidities_three_above + 3*categorical_dummy$comorbidities_two  + 0.01*categorical_dummy$race_White  + creatinine_error 
  
    
    creatinine_mean= mean(creatinine)
    creatinine_sd= sd(creatinine)
    
    # create other continuous variables
    hematocrit = rnorm_pre(creatinine, mu = 32, sd = 1, r = 0.6)
    hematocrit_mean <- mean(hematocrit)
    hematocrit_sd <- sd(hematocrit)
    

    systolic_blood_pressure = rnorm_pre(creatinine, mu = 154, sd = 5, r = 0.5)
    systolic_blood_pressure_mean <- mean(systolic_blood_pressure)
    systolic_blood_pressure_sd <- sd(systolic_blood_pressure)
    
    age = rnorm_pre(creatinine, mu = 60, sd = 8, r = 0.30)
    age_mean <- mean(age)
    age_sd <- sd(age)
    
    
    
    #combine all the variables
    
    continous_outcome_data <- data.frame(categorical_dummy, sex, race, insurance, hematocrit, systolic_blood_pressure, age, creatinine)

    # LR PRECENTAGE should be 22- 58%
    outcome = 320 -0.8*continous_outcome_data$age + 0.15*continous_outcome_data$hematocrit - 0.9*continous_outcome_data$systolic_blood_pressure +0.01*continous_outcome_data$creatinine + 0.001*continous_outcome_data$sex_male -  0.01*continous_outcome_data$race_White - 30*continous_outcome_data$race_Black  -  25*continous_outcome_data$race_Hispanic  - 33*continous_outcome_data$race_Indigenous  +  0.9*continous_outcome_data$insurance_private_insured + 0.2*continous_outcome_data$insurance_public_insured -2*continous_outcome_data$comorbidities_one -5*continous_outcome_data$comorbidities_two -6*continous_outcome_data$comorbidities_three_above + error
    
  
    
    #adding outcome to dataframe
    continous_outcome_all<- data.frame(continous_outcome_data, outcome)
    
    #check the mean outcome of each group
    grp_outcome_mean_df <- continous_outcome_all %>%
      group_by(race) %>%
      summarise_at(vars(outcome), list(outcome_mean = mean))
    
    grp_outcome_mean <- grp_outcome_mean_df$outcome_mean
    #print(grp_outcome_mean_df)
    
    
    #check number of DL cases in each group 
    grp_LR_cases_df <- continous_outcome_all %>%
      group_by(race) %>%
      summarise(LR_cases =sum(outcome < 120))
  
    grp_LR_cases <- grp_LR_cases_df$LR_cases

    m1=lm(outcome~systolic_blood_pressure + creatinine +hematocrit + age+factor(sex) + factor(race) + factor(insurance) + factor(comorbidities), data =continous_outcome_data)
    
  
  i = i + 1

  return(new("Data_Statistics", outcome=outcome, outcome_model=m1, grp_outcome_mean=grp_outcome_mean, grp_LR_cases=grp_LR_cases, systolic_blood_pressure_mean=systolic_blood_pressure_mean, creatinine_mean=creatinine_mean, hematocrit_mean=hematocrit_mean, age_mean=age_mean, systolic_blood_pressure_sd=systolic_blood_pressure_sd, creatinine_sd=creatinine_sd, hematocrit_sd=hematocrit_sd, age_sd=age_sd))


}

outcome_mean_l <- list()
outcome_sd_l <- list()

coeff_intecept <- list()
coeff_B_l <- list()
coeff_H_l <- list()
coeff_In_l <- list()
coeff_W_l <- list()
coeff_systolic_blood_pressure_l <- list()
coeff_creatinine_l <- list()
coeff_hematocrit_l <- list()
coeff_age_l <- list()
coeff_male_l <- list()
coeff_privated_insured_l <- list()
coeff_public_insured_l <- list()
coeff_comorbidities_one_l <- list()
coeff_comorbidities_two_l <- list()
coeff_comorbidities_three_l <- list()

# standard error of coefficients
std_coeff_intecept <- list()
std_coeff_B_l <- list()
std_coeff_H_l <- list()
std_coeff_In_l <- list()
std_coeff_W_l <- list()
std_coeff_systolic_blood_pressure_l <- list()
std_coeff_creatinine_l <- list()
std_coeff_hematocrit_l <- list()
std_coeff_age_l <- list()
std_coeff_male_l <- list()
std_coeff_privated_insured_l <- list()
std_coeff_public_insured_l <- list()
std_coeff_comorbidities_one_l <- list()
std_coeff_comorbidities_two_l <- list()
std_coeff_comorbidities_three_l <- list()

#p value
p_coeff_intecept <- list()
p_coeff_B_l <- list()
p_coeff_H_l <- list()
p_coeff_In_l <- list()
p_coeff_W_l <- list()
p_coeff_systolic_blood_pressure_l <- list()
p_coeff_creatinine_l <- list()
p_coeff_hematocrit_l <- list()
p_coeff_age_l <- list()
p_coeff_male_l <- list()
p_coeff_privated_insured_l <- list()
p_coeff_public_insured_l <- list()
p_coeff_comorbidities_one_l <- list()
p_coeff_comorbidities_two_l <- list()
p_coeff_comorbidities_three_l <- list()


r_square_l <- list()

#mean outcome of each group
Asian_outcome_mean_l <-list()
Black_outcome_mean_l <-list()
Hispanic_outcome_mean_l <-list()
Indigenous_outcome_mean_l <-list()
White_outcome_mean_l <-list()

#late referral cases in each group 
Asian_LR_l <-list()
Black_LR_l <-list()
Hispanic_LR_l <-list()
Indigenous_LR_l <-list()
White_LR_l <-list()

#mean of continous variables
systolic_blood_pressure_mean_l <- list()
creatinine_mean_l <- list()
hematocrit_mean_l <- list()
age_mean_l <- list()

#sd of continous variables
systolic_blood_pressure_sd_l <- list()
creatinine_sd_l <- list()
hematocrit_sd_l <- list()
age_sd_l <- list()


# loop 10000 times
for (i in 1:10000){
  seed <- floor(runif(1, min=1, max=100000))
  i = 0
  
  sub_result = generate_data_4groups(10000, seed)
  outcome_mean_l<- mean(unlist(append(outcome_mean_l, mean(sub_result@outcome))))
  outcome_sd_l<- sd(unlist(append(outcome_sd_l, mean(sub_result@outcome))))
  #outcome correlation 
  
  
  outcome_model <- sub_result@outcome_model
  
  #appending all the coefficients to a list
  coeff_intecept <- append(coeff_intecept, sub_result@outcome_model$coefficients[1])
  coeff_B_l <- append(coeff_B_l, sub_result@outcome_model$coefficients[7])
  coeff_H_l <- append(coeff_H_l, sub_result@outcome_model$coefficients[8])
  coeff_In_l <- append(coeff_In_l, sub_result@outcome_model$coefficients[9])
  coeff_W_l <- append(coeff_W_l, sub_result@outcome_model$coefficients[10])
  coeff_systolic_blood_pressure_l <- append(coeff_systolic_blood_pressure_l, sub_result@outcome_model$coefficients[2])
  coeff_creatinine_l <- append(coeff_creatinine_l, sub_result@outcome_model$coefficients[3])
  coeff_hematocrit_l <- append(coeff_hematocrit_l, sub_result@outcome_model$coefficients[4])
  coeff_age_l <- append(coeff_age_l, sub_result@outcome_model$coefficients[5])
  coeff_male_l <- append(coeff_male_l, sub_result@outcome_model$coefficients[6])
  coeff_privated_insured_l <- append(coeff_privated_insured_l, sub_result@outcome_model$coefficients[11])
  coeff_public_insured_l <- append(coeff_public_insured_l, sub_result@outcome_model$coefficients[12])
  coeff_comorbidities_one_l <- append(coeff_comorbidities_one_l, sub_result@outcome_model$coefficients[13])
  coeff_comorbidities_two_l <- append(coeff_comorbidities_two_l, sub_result@outcome_model$coefficients[15])
  coeff_comorbidities_three_l <- append(coeff_comorbidities_three_l, sub_result@outcome_model$coefficients[14])
  
  r_square_l  <- append(r_square_l, summary(sub_result@outcome_model)$r.squared)[[1]]
  #print(r_square_l)
  
  #standard error of coefficients
 
  std_coeff_intecept <- append(std_coeff_intecept, coef(summary(sub_result@outcome_model))[, "Std. Error"][[1]])
  std_coeff_B_l <- append(std_coeff_B_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][7])
  std_coeff_H_l <- append(std_coeff_H_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][8])
  std_coeff_In_l <- append(std_coeff_In_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][9])
  std_coeff_W_l <- append(std_coeff_W_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][10])
  std_coeff_systolic_blood_pressure_l <- append(std_coeff_systolic_blood_pressure_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][2])
  std_coeff_creatinine_l <- append(std_coeff_creatinine_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][3])
  std_coeff_hematocrit_l <- append(std_coeff_hematocrit_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][4])
  std_coeff_age_l <- append(std_coeff_age_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][5])
  std_coeff_male_l <- append(std_coeff_male_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][6])
  std_coeff_privated_insured_l <- append(std_coeff_privated_insured_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][11])
  std_coeff_public_insured_l <- append(std_coeff_public_insured_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][12])
  std_coeff_comorbidities_one_l <- append(std_coeff_comorbidities_one_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][13])
  std_coeff_comorbidities_two_l <- append(std_coeff_comorbidities_two_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][15])
  std_coeff_comorbidities_three_l <- append(std_coeff_comorbidities_three_l, coef(summary(sub_result@outcome_model))[, "Std. Error"][14])
  
  #p-value
  p_coeff_intecept <- append(p_coeff_intecept, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][[1]])
  p_coeff_B_l <- append(p_coeff_B_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][7])
  p_coeff_H_l <- append(p_coeff_H_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][8])
  p_coeff_In_l <- append(p_coeff_In_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][9])
  p_coeff_W_l <- append(p_coeff_W_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][10])
  p_coeff_systolic_blood_pressure_l <- append(p_coeff_systolic_blood_pressure_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][2])
  p_coeff_creatinine_l <- append(p_coeff_creatinine_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][3])
  p_coeff_hematocrit_l <- append(p_coeff_hematocrit_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][4])
  p_coeff_age_l <- append(p_coeff_age_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][5])
  p_coeff_male_l <- append(p_coeff_male_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][6])
  p_coeff_privated_insured_l <- append(p_coeff_privated_insured_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][11])
  p_coeff_public_insured_l <- append(p_coeff_public_insured_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][12])
  p_coeff_comorbidities_one_l <- append(p_coeff_comorbidities_one_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][13])
  p_coeff_comorbidities_two_l <- append(p_coeff_comorbidities_two_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][15])
  p_coeff_comorbidities_three_l <- append(p_coeff_comorbidities_three_l, coef(summary(sub_result@outcome_model))[, "Pr(>|t|)"][14])
  
  
  
  
  #outcome by group
  Asian_outcome_mean_l <- append(Asian_outcome_mean_l, sub_result@grp_outcome_mean[1])
  Black_outcome_mean_l <- append(Black_outcome_mean_l, sub_result@grp_outcome_mean[2])
  Hispanic_outcome_mean_l <- append(Hispanic_outcome_mean_l, sub_result@grp_outcome_mean[3])
  Indigenous_outcome_mean_l <- append(Indigenous_outcome_mean_l, sub_result@grp_outcome_mean[4])
  White_outcome_mean_l <- append(White_outcome_mean_l, sub_result@grp_outcome_mean[5])
  
  
  #late referal cases by group
  Asian_LR_l <- append(Asian_LR_l, sub_result@grp_LR_cases[1])
  Black_LR_l <- append(Black_LR_l, sub_result@grp_LR_cases[2])
  Hispanic_LR_l <- append(Hispanic_LR_l, sub_result@grp_LR_cases[3])
  Indigenous_LR_l <- append(Indigenous_LR_l, sub_result@grp_LR_cases[4])
  White_LR_l <- append(White_LR_l, sub_result@grp_LR_cases[5])
  
  #continous varibles mean 
  systolic_blood_pressure_mean_l <- append(systolic_blood_pressure_mean_l, sub_result@systolic_blood_pressure_mean)
  creatinine_mean_l <- append(creatinine_mean_l, sub_result@creatinine_mean)
  hematocrit_mean_l <- append(hematocrit_mean_l, sub_result@hematocrit_mean)
  age_mean_l <- append(age_mean_l, sub_result@age_mean)
  
  #continous varibles sd
  systolic_blood_pressure_sd_l <- append(systolic_blood_pressure_sd_l, sub_result@systolic_blood_pressure_sd)
  creatinine_sd_l <- append(creatinine_sd_l, sub_result@creatinine_sd)
  hematocrit_sd_l <- append(hematocrit_sd_l, sub_result@hematocrit_sd)
  age_sd_l <- append(age_sd_l, sub_result@age_sd)
}

#print
mean(outcome_mean_l)
mean(outcome_mean_l)

# mean of coefficients
print(mean(unlist(coeff_intecept)))
print(mean(unlist(coeff_B_l)))
print(mean(unlist(coeff_H_l)))
print(mean(unlist(coeff_In_l)))
print(mean(unlist(coeff_W_l)))
print(mean(unlist(coeff_systolic_blood_pressure_l)))
print(mean(unlist(coeff_creatinine_l)))
print(mean(unlist(coeff_hematocrit_l)))
print(mean(unlist(coeff_age_l)))
print(mean(unlist(coeff_male_l)))
print(mean(unlist(coeff_privated_insured_l)))
print(mean(unlist(coeff_public_insured_l)))
print(mean(unlist(coeff_comorbidities_one_l)))
print(mean(unlist(coeff_comorbidities_two_l)))
print(mean(unlist(coeff_comorbidities_three_l)))


#mean outcome of each group
print(mean(unlist(Asian_outcome_mean_l)))
print(mean(unlist(Black_outcome_mean_l))) 
print(mean(unlist(Hispanic_outcome_mean_l)))
print(mean(unlist(Indigenous_outcome_mean_l)))
print(mean(unlist(White_outcome_mean_l)))

#mean late referal cases of each group
print(mean(unlist(Asian_LR_l)))
print(mean(unlist(Black_LR_l)))
print(mean(unlist(Hispanic_LR_l))) 
print(mean(unlist(Indigenous_LR_l)))
print(mean(unlist(White_LR_l)))

#mean of continous variables
print(mean(unlist(systolic_blood_pressure_mean_l)))
print(mean(unlist(creatinine_mean_l)))
print(mean(unlist(hematocrit_mean_l)))
print(mean(unlist(age_mean_l)))

#sd of continous variables
print(mean(unlist(systolic_blood_pressure_sd_l)))
print(mean(unlist(creatinine_sd_l)))
print(mean(unlist(hematocrit_sd_l)))
print(mean(unlist(age_sd_l))) 

#r square
print(mean(r_square_l))

#coefficent standard error
print(mean(unlist(std_coeff_intecept)))
print(mean(unlist(std_coeff_B_l)))
print(mean(unlist(std_coeff_H_l)))
print(mean(unlist(std_coeff_In_l)))
print(mean(unlist(std_coeff_W_l)))
print(mean(unlist(std_coeff_systolic_blood_pressure_l)))
print(mean(unlist(std_coeff_creatinine_l)))
print(mean(unlist(std_coeff_hematocrit_l)))
print(mean(unlist(std_coeff_age_l)))
print(mean(unlist(std_coeff_male_l)))
print(mean(unlist(std_coeff_privated_insured_l)))
print(mean(unlist(std_coeff_public_insured_l)))
print(mean(unlist(std_coeff_comorbidities_one_l)))
print(mean(unlist(std_coeff_comorbidities_two_l)))
print(mean(unlist(std_coeff_comorbidities_three_l)))


#p value
print(mean(unlist(p_coeff_intecept)))
print(mean(unlist(p_coeff_B_l)))
print(mean(unlist(p_coeff_H_l)))
print(mean(unlist(p_coeff_In_l)))
print(mean(unlist(p_coeff_W_l)))
print(mean(unlist(p_coeff_systolic_blood_pressure_l)))
print(mean(unlist(p_coeff_creatinine_l)))
print(mean(unlist(p_coeff_hematocrit_l)))
print(mean(unlist(p_coeff_age_l)))
print(mean(unlist(p_coeff_male_l)))
print(mean(unlist(p_coeff_privated_insured_l)))
print(mean(unlist(p_coeff_public_insured_l)))
print(mean(unlist(p_coeff_comorbidities_one_l)))
print(mean(unlist(p_coeff_comorbidities_two_l)))
print(mean(unlist(p_coeff_comorbidities_three_l)))

