####################################################################################
# The e-mail we should send (around) a week before the course starts
############################################################################

Dear students,

the specialist course "Introduction to Bayesian Statistical Modelling" will start next week. During this course, we recommand you to use your personal laptop, on which you will have installed R (version 3.4.3). We decided to use the RStudio user interface. To facilitate our interactions, we recommand use installing it as well: https://www.rstudio.com/products/rstudio/

We will also use Stan, from RStudio, using the package rstan. Please find below detailed instructions on how to install it according to your operating system: https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started

We will use the following R packages, that can be installed directly from RStudio, copy-pasting the following command in the console:

install.packages(c("data.table","coda","mvtnorm","devtools","MASS","ellipse","rstan","coda","markdown","mcmc","MCMCpack","MuMIn","reshape2","rmarkdown","brms","tidyverse","bayesplot","shinystan","lme4"), dependencies = TRUE)

Slides, data and scripts will be available on Github (from the first day of the course) following this link: https://github.com/lnalborczyk/Bayes_UGent_2018

Although a good knowledge of R was a prerequisite of registration, please find atached a short introduction to the R language, which will be necessary to understand the examples discussed during the course.

Please find below an updated version of the planning, and accept our apologies concerning the discrepancies between this planning and the initial planning.

Day 1 (Wednesday 13 June)

09h00-10h30: Introduction to the tidyverse
10h30-12h00: The model comparison approach to data analysis
12h00-13h00: Lunch break
13h00-14h30: Model comparison and information criteria
14h30-16h00: Introduction to Bayesian Inference

Day 2 (Thursday 14 June)

09h00-12h00: Modelling of continuous data via linear regression
12h00-13h00: Lunch break
13h00-16h00: Generalized linear models (count and percentage data, Likert data)

Day 3 (Friday 15 June)

09h00-10h30: Analysing response times
10h30-12h00: Bayesian meta-analysis using brms
12h00-13h00: Lunch break
13h00-14h30: Non-linear models
14h30-16h00: Measurement error, handling missing values

Best regards,

The organising committee
