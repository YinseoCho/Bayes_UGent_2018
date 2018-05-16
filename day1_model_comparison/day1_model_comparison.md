Model comparison and information criteria
========================================================
author: Ladislas Nalborczyk
date: Univ. Grenoble Alpes, CNRS, LPNC (France) • Ghent University (Belgium)
autosize: true
transition: none
width: 1600
height: 1000
css: css-file.css

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: true
type: lineheight



<!-- For syntax highlighting -->
<link rel="stylesheet" href="github.css">

Let's say we are interested in height differences between women and men...


```r
men <- rnorm(100, 175, 10) # 100 men heights
women <- rnorm(100, 170, 10) # 100 women heights
```


```r
t.test(men, women)
```

```

	Welch Two Sample t-test

data:  men and women
t = 3.8455, df = 195.5, p-value = 0.0001627
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 2.716070 8.434909
sample estimates:
mean of x mean of y 
 177.0399  171.4644 
```

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: true
type: lineheight

We are going to simulate t-values computed on samples generated under the assumption of no difference between women and men (the null hypothesis H0).


```r
nSims <- 1e4 # number of simulations
t <- rep(NA, nSims) # initialising an empty vector

for (i in 1:nSims) {
    
    men2 <- rnorm(100, 170, 10)
    women2 <- rnorm(100, 170, 10)
    t[i] <- t.test(men2, women2)$statistic
    
}
```


```r
t <- replicate(nSims, t.test(rnorm(100, 170, 10), rnorm(100, 170, 10) )$statistic)
```

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: false
type: lineheight


```r
data.frame(t = t) %>%
    ggplot(aes(x = t) ) +
    geom_histogram() +
    theme_bw(base_size = 20)
```

<img src="day1_model_comparison-figure/unnamed-chunk-5-1.png" title="plot of chunk unnamed-chunk-5" alt="plot of chunk unnamed-chunk-5" style="display: block; margin: auto;" />

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: false
type: lineheight


```r
data.frame(t = c(-5, 5) ) %>%
    ggplot(aes(x = t) ) +
    stat_function(fun = dt, args = list(df = t.test(men, women)$parameter), size = 1.5) +
    theme_bw(base_size = 20) + ylab("density")
```

<img src="day1_model_comparison-figure/unnamed-chunk-6-1.png" title="plot of chunk unnamed-chunk-6" alt="plot of chunk unnamed-chunk-6" style="display: block; margin: auto;" />

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: false
type: lineheight


```r
alpha <- .05
abs(qt(alpha / 2, df = t.test(men, women)$parameter) ) # two-sided critical t-value
```

```
[1] 1.972173
```

<img src="day1_model_comparison-figure/unnamed-chunk-8-1.png" title="plot of chunk unnamed-chunk-8" alt="plot of chunk unnamed-chunk-8" style="display: block; margin: auto;" />

Null Hypothesis Significance Testing (NHST)
========================================================
incremental: false
type: lineheight


```r
tobs <- t.test(men, women)$statistic # observed t-value
tobs %>% as.numeric
```

```
[1] 3.845475
```

<img src="day1_model_comparison-figure/unnamed-chunk-10-1.png" title="plot of chunk unnamed-chunk-10" alt="plot of chunk unnamed-chunk-10" style="display: block; margin: auto;" />

P-values
========================================================
incremental: true
type: lineheight

A p-value is simply a tail area (an integral), under the distribution of test statistics under the null hypothesis. It gives the probability of observing the data we observed (or more extreme data), **given that the null hypothesis is true**.

$$p[\mathbf{t}(\mathbf{x}^{\text{rep}}|H_{0}) \geq t(x)]$$


```r
t.test(men, women)$p.value
```

```
[1] 0.0001627272
```

```r
tvalue <- abs(t.test(men, women)$statistic)
df <- t.test(men, women)$parameter

2 * integrate(dt, tvalue, Inf, df = df)$value
```

```
[1] 0.0001627272
```

Model comparison
========================================================
incremental: true
type: lineheight

Two common problems in statistical learning: overfitting and underfitting... how to avoid it ?

- Using regularisation. The aim is to constrain the learning, to constrain the influence of the incoming data on the inference.
- Using cross-validation or information criteria (e.g., AIC, WAIC)

<br>

$$R^{2} = \frac{\text{var}(\text{outcome}) - \text{var}(\text{residuals})}{\text{var}(\text{outcome})} =
1 - \frac{\text{var}(\text{residuals})}{\text{var}(\text{outcome})}$$

Overfitting
========================================================
incremental: false
type: lineheight

<img src="day1_model_comparison-figure/unnamed-chunk-12-1.png" title="plot of chunk unnamed-chunk-12" alt="plot of chunk unnamed-chunk-12" style="display: block; margin: auto;" />

Overfitting
========================================================
incremental: true
type: lineheight


```r
mod1.1 <- lm(brain ~ mass, data = d)
(var(d$brain) - var(residuals(mod1.1) ) ) / var(d$brain)
```

```
[1] 0.490158
```

```r
mod1.2 <- lm(brain ~ mass + I(mass^2), data = d)
(var(d$brain) - var(residuals(mod1.2) ) ) / var(d$brain)
```

```
[1] 0.5359967
```

```r
mod1.3 <- lm(brain ~ mass + I(mass^2) + I(mass^3), data = d)
(var(d$brain) - var(residuals(mod1.3) ) ) / var(d$brain)
```

```
[1] 0.6797736
```



Overfitting
========================================================
incremental: false
type: lineheight

<img src="day1_model_comparison-figure/unnamed-chunk-15-1.png" title="plot of chunk unnamed-chunk-15" alt="plot of chunk unnamed-chunk-15" style="display: block; margin: auto;" />

Underfitting
========================================================
incremental: false
type: lineheight

$$\begin{align}
v_{i} &\sim \mathrm{Normal}(\mu_{i}, \sigma) \\
\mu_{i} &= \alpha \\
\end{align}$$


```r
mod1.7 <- lm(brain ~ 1, data = d)
```

<img src="day1_model_comparison-figure/unnamed-chunk-17-1.png" title="plot of chunk unnamed-chunk-17" alt="plot of chunk unnamed-chunk-17" style="display: block; margin: auto;" />

Information theory
========================================================
incremental: true
type: lineheight

We would like to mesure the *distance* between our model and the *full reality* (i.e., the data generating process)... but we will first make a detour via information theory.

We can define **information** as the amount of reduction in uncertainty. When we learn a new outcome (a new observation), how much does it reduce our uncertainty ?

We need a way a measuring uncertainty. For $n$ possible events, with each event $i$ having a probability $p_{i}$, a measure of uncertainty is given by:

$$H(p) = - \text{E log}(p_{i}) = - \sum_{i=1}^{n}p_{i} \text{log}(p_{i})$$

In other words, *the uncertainty contained in a probability distribution is the average log-probability of an event*.

Uncertainty
========================================================
incremental: true
type: lineheight

Let's take as an example weather forecasting. Let's say the probability of having rain or sun on an average day in Ghent is, respectively, $p_{1} = 0.7$ and $p_{2} = 0.3$.

Then, $H(p) = - (p_{1} \text{log}(p_{1}) + p_{2} \text{log}(p_{2}) ) \approx 0.61$.


```r
p <- c(0.7, 0.3)
- sum(p * log(p) )
```

```
[1] 0.6108643
```

Now let's consider the weather in Abu Dabi. There, the probability of having rain or sun is of $p_{1} = 0.01$ and $p_{2} = 0.99$.


```r
p <- c(0.01, 0.99)
- sum(p * log(p) )
```

```
[1] 0.05600153
```

Divergence
========================================================
incremental: true
type: lineheight

We now have a way of quantifying uncertainty. How does it help us to measure the distance between our model and the full reality ?

**Divergence**: uncertainty added by using a probability distribution to describe... another probability distribution ([Kullback-Leibler divergence](https://fr.wikipedia.org/wiki/Divergence_de_Kullback-Leibler)).

$$D_{KL}(p,q) = \sum_{i} p_{i}\big(\text{log}(p_{i}) - \text{log}(q_{i})\big) = \sum_{i} p_{i} \text{log}\bigg(\frac{p_{i}}{q_{i}}\bigg)$$

Divergence
========================================================
incremental: true
type: lineheight

$$D_{KL}(p,q) = \sum_{i} p_{i}\big(\text{log}(p_{i}) - \text{log}(q_{i})\big) = \sum_{i} p_{i} \text{log}\bigg(\frac{p_{i}}{q_{i}}\bigg)$$

As an example, let's say that the *true* probability of the events *rain* and *sun* is $p_{1} = 0.3$ and $p_{2} = 0.7$. What uncertainty do we add if we think that the probabilities are rather $q_{1} = 0.25$ and $q_{2} = 0.75$ ?


```r
p <- c(0.3, 0.7)
q <- c(0.25, 0.75)

sum(p * log(p / q) )
```

```
[1] 0.006401457
```

```r
sum(q * log(q / p) )
```

```
[1] 0.006164264
```

Cross entropy and Divergence
========================================================
incremental: true
type: lineheight

**Cross entropy**: $H(p,q) = \sum_{i} p_{i} \log (q_{i})$


```r
sum(p * (log(q) ) )
```

The **Divergence** can be defined as the additional entropy added when using $q$ to describe $p$.

$$
\begin{align}
D_{KL}(p,q) &= H(p,q) - H(p) \\
&= - \sum_{i} p_{i} \log(q_{i}) - \big( - \sum_{i} p_{i} \log(p_{i}) \big) \\
&= - \sum_{i} p_{i} \big(\log(q_{i}) - \log(p_{i}) \big) \\ 
\end{align}
$$


```r
- sum (p * (log(q) - log(p) ) )
```

```
[1] 0.006401457
```

Toward the deviance...
========================================================
incremental: false
type: lineheight

Fine. But we do not know the full reality in real life...

We do not need to know it ! When comparing two models (two distributions) $q$ and $r$, to approximate $p$, we can compare their divergences. Thus, $\text{E} \ \text{log}(p_{i})$ will be the same quantity for both divergences... !

<div align = "center" style="border:none;">
<img src = "mind_blowing.jpg" width = 400 height = 400>
</div>

Toward the deviance...
========================================================
incremental: false
type: lineheight

We can use $\text{E} \ \text{log}(q_{i})$ and $\text{E} \ \text{log}(r_{i})$ as estimations of the relative distance between each model end the target distribution (the full reality). We only need the avearge log-probability of the model.

As we do not know the target distribution, we can not interpret these values in absolute terms. We are interested in the relative distances: $\text{E} \ \text{log}(q_{i}) - \text{E} \ \text{log}(r_{i})$.

<div align = "center" style="border:none;">
<img src = "KL_distance.png" width = 286 height = 547>
</div>

Deviance
========================================================
incremental: true
type: lineheight

To approximate $\text{E} \ \log(p_{i})$, we use the [deviance](https://en.wikipedia.org/wiki/Deviance_%28statistics%29) of a model, whic measures "how bad" is a model to explain some data.

$$D(q) = -2 \sum_{i} \log(q_{i})$$

where $i$ indexes observations and $q_{i}$ is the *likelihood* of each observation.


```r
d$mass.s <- scale(d$mass)
mod1.8 <- lm(brain ~ mass.s, data = d)

-2 * logLik(mod1.8) # computing the deviance
```

```
'log Lik.' 94.92499 (df=3)
```

Deviance
========================================================
incremental: true
type: lineheight


```r
# extracting model's coefficients

alpha <- coef(mod1.8)[1]
beta <- coef(mod1.8)[2]

# computing the log-likelihood

ll <-
    sum(
        dnorm(
            d$brain,
            mean = alpha + beta * d$mass.,
            sd = sd(residuals(mod1.8) ),
            log = TRUE
            )
        )

# computing the deviance

(-2) * ll
```

```
[1] 95.00404
```

In-sample and out-of-sample
========================================================
incremental: true
type: lineheight

The deviance has the same problem as the $R^{2}$, when it is computed on the training dataset (the observed data). In this situation, we call it **in-sample deviance**.

If we are interested in the predictive abilities of a model, we can compute the deviance of the model on a new dataset (the test dataset)... We call this the **out-of-sample deviance**. It answers the question: how bad is the model to predict future data ?

Let's say we have a sample of size $N$, that we call the *training dataset*. We can compute the deviance of a model on this sample ($D_{train}$ or $D_{in}$). If we then acquire a new sample of size $N$ issued from the same data generating process (we call it the *test dataset*), we can compute the deviance of the model on this new dataset, by using the values of the parameters we estimated with the training dataset (we call this $D_{test}$ or $D_{out}$).

In sample and out of sample deviance
========================================================
incremental: true
type: lineheight

$$
\begin{align}
y_{i} &\sim \mathrm{Normal}(\mu_{i},1) \\
\mu_{i} &= (0.15) x_{1,i} - (0.4) x_{2,i} \\
\end{align}
$$

<div align = "center" style="border:none;">
<img src = "inout1.png" width = 1000 height = 500>
</div>

We generated data from the above model 10.000 times, and computed the in-sample and out-of-sample deviance of 5 linear models of increasing complexity.

Regularisation
========================================================
incremental: false
type: lineheight

Another way to fight overfitting is to use skeptical priors that will prevent the model to learn *too much* from the data. In other words, we can use a stronger prior in order to diminish the weight of the data.

<img src="day1_model_comparison-figure/unnamed-chunk-25-1.png" title="plot of chunk unnamed-chunk-25" alt="plot of chunk unnamed-chunk-25" style="display: block; margin: auto;" />

Regularisation and cross-validation
========================================================
incremental: true
type: lineheight

<div align = "center" style="border:none;">
<img src = "inout2.png" width = 1200 height = 600>
</div>

How to decide on the widht of a prior ? How to know whether a prior is *sufficiently regularising* or not ? We can divide the dataset in two parts (training and test) in order to compare different priors. We can then choose the prior that provides the lower **out-of-sample deviance**. We call this strategy **cross-validation**.

Out-of-sample deviance and information criteria
========================================================
incremental: false
type: lineheight

<div align = "center" style="border:none;">
<img src = "inout3.png" width = 1200 height = 600>
</div>

We notice that the out-of-sample deviance is approximately equal to the in-sample deviance, plus two times the number of parameters of the model...

Akaike information criterion
========================================================
incremental: true
type: lineheight

The AIC offers an approximation of the **out-of-sample deviance** as:

$$\text{AIC} = D_{train} + 2p$$

where $p$ is the number of parameters of the model. The AIC then gives an approximation of the relative **predictive abilities** of models.

NB: when the number of observations $N$ is low as compared to the number of parameters $p$ of the model (e.g., when $N / p> 40$), the second-order correction of the AIC should be used (e.g., see Burnham & Anderson, [2002](https://www.springer.com/us/book/9780387953649); [2004](http://journals.sagepub.com/doi/abs/10.1177/0049124104268644)).

Akaike weights and evidence ratios
========================================================
incremental: true
type: lineheight

The individual AIC values are not interpretable in absolute terms as they contain arbitrary constants. We usually rescale them by substracting to the AIC of each model $i$ the AIC of the model with the minimum one:

$$\Delta_{AIC} = AIC_{i} - AIC_{min}$$

This transformation forces the best model to have $\Delta = 0$, while the rest of the models have positive values. Then, the simple transformation $exp(-\Delta_{i}/2)$ provides the likelihood of the model given the data $\mathcal{L}(g_{i}|data)$ ([Akaike, 1981](https://www.sciencedirect.com/science/article/pii/0304407681900713)).

Akaike weights and evidence ratios
========================================================
incremental: true
type: lineheight

It is convenient to normalise the model likelihoods such that they sum to 1 and that we can treat them as probabilities. Hence, we use:

$$w_{i} = \dfrac{exp(-\Delta_{i}/2)}{\sum_{r = 1}^{R}exp(-\Delta_{r}/2)}$$

The weights $w_{i}$ are useful as the *weight of evidence* in favour of model $g_{i}$ as being the actual best model in the set of models, in an information-theretical sense (i.e., the closest model to the *truth*).

From there, we can compute evidence ratios (ERs) as the ratios of weights: $ER_{ij} = \frac{w_{i}}{w_{j}}$, where $w_{i}$ and $w_{j}$ are the Akaike weights of models $i$ and $j$, respectively.

Akaike weights and evidence ratios
========================================================
incremental: true
type: lineheight

We want to predict how many kilo calories per gram of milk (`kcal.per.g`) different species do have, as a function of neocortex volume and body mass. We can fit a few models and compare them using the AIC.


```r
library(rethinking)
data(milk)

d <-
    milk %>%
    na.omit %>%
    mutate(neocortex = neocortex.perc / 100)

head(d, 10)
```

```
              clade             species kcal.per.g perc.fat perc.protein
1     Strepsirrhine      Eulemur fulvus       0.49    16.60        15.42
2  New World Monkey  Alouatta seniculus       0.47    21.22        23.58
3  New World Monkey          A palliata       0.56    29.66        23.46
4  New World Monkey        Cebus apella       0.89    53.41        15.80
5  New World Monkey          S sciureus       0.92    50.58        22.33
6  New World Monkey    Cebuella pygmaea       0.80    41.35        20.85
7  New World Monkey   Callimico goeldii       0.46     3.93        25.30
8  New World Monkey  Callithrix jacchus       0.71    38.38        20.09
9  Old World Monkey Miopithecus talpoin       0.68    40.15        18.08
10 Old World Monkey           M mulatta       0.97    55.51        13.17
   perc.lactose mass neocortex.perc neocortex
1         67.98 1.95          55.16    0.5516
2         55.20 5.25          64.54    0.6454
3         46.88 5.37          64.54    0.6454
4         30.79 2.51          67.64    0.6764
5         27.09 0.68          68.85    0.6885
6         37.80 0.12          58.85    0.5885
7         70.77 0.47          61.69    0.6169
8         41.53 0.32          60.32    0.6032
9         41.77 1.55          69.97    0.6997
10        31.32 3.24          70.41    0.7041
```

Akaike weights and evidence ratios
========================================================
incremental: true
type: lineheight


```r
m1 <- lm(kcal.per.g ~ 1, data = d, na.action = na.fail)
m2 <- lm(kcal.per.g ~ 1 + neocortex, data = d, na.action = na.fail)
m3 <- lm(kcal.per.g ~ 1 + log(mass), data = d, na.action = na.fail)
m4 <- lm(kcal.per.g ~ 1 + neocortex + log(mass), data = d, na.action = na.fail)
```


```r
library(MuMIn) # package for MultiModel Inference

(ictab <- model.sel(list(m1 = m1, m2 = m2, m3 = m3, m4 = m4) ) )
```

```
Model selection table 
     (Int)    ncr log(mss) df logLik  AICc delta weight
m4 -1.0850 2.7930 -0.09640  4 12.678 -14.0  0.00  0.926
m1  0.6576                  2  6.229  -7.6  6.42  0.037
m3  0.7052        -0.03169  3  7.369  -6.9  7.13  0.026
m2  0.3533 0.4503           3  6.437  -5.0  8.99  0.010
Models ranked by AICc(x) 
```

Model averaging
========================================================
incremental: true
type: lineheight


```r
mavg <- model.avg(ictab)
summary(mavg)
```

```

Call:
model.avg(object = ictab)

Component model call: 
lm(formula = <4 unique values>, data = d, na.action = na.fail)

Component models: 
       df logLik   AICc delta weight
12      4  12.68 -14.02  0.00   0.93
(Null)  2   6.23  -7.60  6.42   0.04
2       3   7.37  -6.89  7.13   0.03
1       3   6.44  -5.03  8.99   0.01

Term codes: 
neocortex log(mass) 
        1         2 

Model-averaged coefficients:  
(full average) 
            Estimate Std. Error Adjusted SE z value Pr(>|z|)   
(Intercept) -0.95838    0.67184     0.70741   1.355  0.17549   
neocortex    2.59136    1.05470     1.10947   2.336  0.01951 * 
log(mass)   -0.09011    0.03306     0.03474   2.594  0.00949 **
 
(conditional average) 
            Estimate Std. Error Adjusted SE z value Pr(>|z|)   
(Intercept) -0.95838    0.67184     0.70741   1.355  0.17549   
neocortex    2.76725    0.83735     0.90980   3.042  0.00235 **
log(mass)   -0.09462    0.02684     0.02899   3.264  0.00110 **
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Relative variable importance: 
                     log(mass) neocortex
Importance:          0.95      0.94     
N containing models:    2         2     
```

Model averaging
========================================================
incremental: false
type: lineheight

We can then plot the predictions of each model and the averaged predictions (in black, along with its 95% CIs),  according to `neocortex` and for an average value of `mass`.

<img src="day1_model_comparison-figure/unnamed-chunk-30-1.png" title="plot of chunk unnamed-chunk-30" alt="plot of chunk unnamed-chunk-30" style="display: block; margin: auto;" />

Practice - 1/3
========================================================
incremental: false
type: lineheight

Using the attitude `dataset` explored previously...


```r
data <- read.csv("attitude.csv")
head(data)
```

```
  participant    sex drink  imagery ratings
1           1 female  beer negative       6
2           2 female  beer negative      30
3           3 female  beer negative      15
4           4 female  beer negative      30
5           5 female  beer negative      12
6           6 female  beer negative      17
```

Practice - 2/3
========================================================
incremental: false
type: lineheight

Build the following linear models, using the `lm` function...

$$
\begin{align}

\mathcal{M_{1}} : \text{ratings}_{i} &\sim \mathrm{Normal}(\mu_{i},\sigma) \\
\mu_{i} &= \alpha + \beta_{1} \text{sex}_{i} \\

\mathcal{M_{2}} : \text{ratings}_{i} &\sim \mathrm{Normal}(\mu_{i},\sigma) \\
\mu_{i} &= \alpha + \beta_{1} \text{sex}_{i} + \beta_{2} \text{drink}_{i} \\

\mathcal{M_{3}} : \text{ratings}_{i} &\sim \mathrm{Normal}(\mu_{i},\sigma) \\
\mu_{i} &= \alpha + \beta_{1} \text{sex}_{i} + \beta_{2} \text{drink}_{i} + \beta_{3} \text{imagery}_{i} \\

\mathcal{M_{4}} : \text{ratings}_{i} &\sim \mathrm{Normal}(\mu_{i},\sigma) \\
\mu_{i} &= \alpha + \beta_{1} \text{drink}_{i} + \beta_{2} \text{imagery}_{i} \\

\mathcal{M_{5}} : \text{ratings}_{i} &\sim \mathrm{Normal}(\mu_{i},\sigma) \\
\mu_{i} &= \alpha + \beta_{1} \text{sex}_{i} + \beta_{2} \text{imagery}_{i} \\

\end{align}
$$

Practice - 3/3
========================================================
incremental: false
type: lineheight

1. Compare these models using the AIC. Compare the ranks and the weights of the models.

2. For each model, plot the estimated mean and 90% confidence interval of the mean, overlaid on the raw data. How the predictions differ across the models ?

3. Make a plot of the averaged predictions (averaged on the three best models). How these averaged predictions differ from the predictions of the best model (i.e., the model with the lowest AIC) ?

4. Compute the out-of-sample deviance of each model.

5. Compare the out-of-sample deviance values computed above to the AIC values. Based on the deviance values, which model makes the best predictions ? Is the AIC a good estimator of the out-of-sample deviance ?

Solution - question 1
========================================================
incremental: false
type: lineheight


```r
m1 <- lm(ratings ~ sex, data)
m2 <- lm(ratings ~ drink, data)
m3 <- lm(ratings ~ drink + imagery + drink:imagery, data)
m4 <- lm(ratings ~ sex + drink + imagery + drink:imagery, data)
m5 <- lm(ratings ~ sex + drink + imagery, data)

(ictab <- model.sel(list(m1 = m1, m2 = m2, m3 = m3, m4 = m4, m5 = m5) ) )
```

```
Model selection table 
   (Int) sex drn img drn:img df   logLik  AICc delta weight
m5 7.325   +   +   +          5 -298.315 607.4  0.00  0.567
m4 8.825   +   +   +       +  6 -297.418 608.0  0.55  0.432
m3 4.450       +   +       +  5 -304.472 619.8 12.32  0.001
m2 7.225       +              3 -310.868 628.1 20.61  0.000
m1 6.275   +                  3 -313.437 633.2 25.75  0.000
Models ranked by AICc(x) 
```

Solution - question 2
========================================================
incremental: false
type: lineheight


```r
n.trials <- 1e4
age.seq <- seq(from = -2, to = 3.5, length.out = 58)
prediction.data <- data.frame(age = age.seq)

computeMu <- function(model, data, n.trials) {
    
    mu <- link(fit = model, data = data, n = n.trials)
    return(mu)

}

computeMuMean <- function(mu) {
    
    mu.mean <- apply(X = mu, MARGIN = 2, FUN = mean)
    return(mu.mean)
    
}

computeMuHPDI <- function(mu) {
    
    mu.HPDI <- apply(X = mu, MARGIN = 2, FUN = HPDI, prob = 0.97)
    return(mu.HPDI)
    
}
```

Solution - question 2
========================================================
incremental: false
type: lineheight


```r
simulateHeights <- function(model, prediction.data) {
    
    simulated.heights <- sim(fit = model, data = prediction.data)
    return(simulated.heights)
    
}

plotResults <- function(model, prediction.data, original.data, n.trials) {
    
    mu <- computeMu(model, prediction.data, n.trials)
    mu.mean <- computeMuMean(mu)
    mu.HPDI <- computeMuHPDI(mu)
    simulated.heights <- simulateHeights(model = model, prediction.data = prediction.data)
    simulated.heights.HPDI <- apply(X = simulated.heights, MARGIN = 2, FUN = HPDI)
    plot(height ~ age, data = original.data, col = "steelblue", pch = 16)
    lines(x = prediction.data$age, y = mu.mean, lty = 2)
    lines(x = prediction.data$age, y = mu.HPDI[1, ], lty = 2)
    lines(x = prediction.data$age, y = mu.HPDI[2, ], lty = 2)
    shade(object = simulated.heights.HPDI, lim = prediction.data$age)
    
}
```

Solution - question 2
========================================================
incremental: false
type: lineheight


```r
plotResults(
    model = mod3.1, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)

plotResults(
    model = mod3.2, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)

plotResults(
    model = mod3.3, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)

plotResults(
    model = mod3.4, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)

plotResults(
    model = mod3.5, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)

plotResults(
    model = mod3.6, prediction.data = prediction.data,
    original.data = d1, n.trials = n.trials)
```

Solution - question 3
========================================================
incremental: false
type: lineheight


```r
h.ensemble <- ensemble(mod3.4, mod3.5, mod3.6, data = list(age = age.seq) )
mu.mean <- apply(h.ensemble$link, 2, mean)
mu.ci <- t(apply(h.ensemble$link, 2, HPDI) )
height.ci <- t(apply(h.ensemble$sim, 2, HPDI) )

ggplot(data = d1, aes(x = as.numeric(age), y = height) ) +
    geom_point(size = 2) +
    geom_line(data = data.frame(age = age.seq, height = mu.mean) ) +
    geom_ribbon(
        data = data.frame(mu.ci), inherit.aes = FALSE,
        aes(x = age.seq, ymin = mu.ci[, 1], ymax = mu.ci[, 2]), alpha = 0.2) +
    geom_ribbon(
        data = data.frame(height.ci), inherit.aes = FALSE,
        aes(x = age.seq, ymin = height.ci[, 1], ymax = height.ci[, 2]), alpha = 0.1) +
    theme_bw(base_size = 20) + xlim(-2, 3) + ylim(60, 190)
```

Solution - question 4
========================================================
incremental: false
type: lineheight


```r
# model 1
coefs <- coef(mod3.1)
mu <- coefs["alpha"] + coefs["beta.1"] * d2$age
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.1 <- -2 * log.likelihood

# model 2
coefs <- coef(mod3.2)
mu <- coefs["alpha"] + coefs["beta.1"] * d2$age + coefs["beta.2"] * (d2$age)^2
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.2 <- -2 * log.likelihood

# model 3
coefs <- coef(mod3.3)
mu <- coefs["alpha"] + coefs["beta.1"] * d2$age + coefs["beta.2"] * (d2$age)^2 + coefs["beta.3"] * (d2$age)^3
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.3 <- -2 * log.likelihood
```

Solution - question 4
========================================================
incremental: false
type: lineheight


```r
# model 4
coefs <- coef(mod3.4)
mu <- coefs["alpha"] + coefs["beta.1"]*d2$age + coefs["beta.2"]*(d2$age)^2 + coefs["beta.3"]*(d2$age)^3 + coefs["beta.4"]*(d2$age)^4
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.4 <- -2 * log.likelihood

# model 5
coefs <- coef(mod3.5)
mu <- coefs["alpha"] + coefs["beta.1"]*d2$age + coefs["beta.2"]*(d2$age)^2 + coefs["beta.3"]*(d2$age)^3 + coefs["beta.4"]*(d2$age)^4 + coefs["beta.5"]*(d2$age)^5
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.5 <- -2 * log.likelihood

# model 6
coefs <- coef(mod3.6)
mu <- coefs["alpha"] + coefs["beta.1"]*d2$age + coefs["beta.2"]*(d2$age)^2 + coefs["beta.3"]*(d2$age)^3 + coefs["beta.4"]*(d2$age)^4 + coefs["beta.5"]*(d2$age)^5 + coefs["beta.6"]*(d2$age)^6
log.likelihood <- sum(dnorm(x = d2$height, mean = mu, sd = coefs["sigma"], log = TRUE) )
dev.mod3.6 <- -2 * log.likelihood
```

Solution - question 5
========================================================
incremental: false
type: lineheight


```r
deviances <- c(dev.mod3.1,dev.mod3.2,dev.mod3.3,dev.mod3.4,dev.mod3.5,dev.mod3.6)
comparison <- compare(mod3.1,mod3.2,mod3.3,mod3.4,mod3.5,mod3.6)
comparison <- as.data.frame(comparison@output)
comparison <- comparison[order(rownames(comparison) ), ]
waics <- comparison$WAIC

data.frame(deviance = deviances, waic = waics) %>%
    gather(type, value) %>%
    mutate(x = rep(1:6, 2) ) %>%
    ggplot(aes(x = x, y = value, colour = type) ) +
    scale_colour_grey() +
    geom_point(size = 2) +
    scale_x_continuous(breaks = 1:6) +
    theme_bw(base_size = 20) + xlab("model") + ylab("Déviance/WAIC")
```

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
<script>

for(i=0;i<$("section").length;i++) {
if(i==0) continue
$("section").eq(i).append("<p style='font-size:xx-large;position:fixed;right:200px;bottom:50px;'>" + i + "</p>")
}

</script>
