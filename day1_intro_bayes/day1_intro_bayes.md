Introduction to Bayesian inference
========================================================
author: Ladislas Nalborczyk
date: Univ. Grenoble Alpes, CNRS, LPNC (France) • Ghent University (Belgium)
autosize: true
transition: none
width: 1600
height: 1000
css: css-file.css

Probabilistic interpretations
========================================================
incremental: true
type: lineheight



<!-- For syntax highlighting -->
<link rel="stylesheet" href="github.css">

What is the probability...

* Of obtaining an even number when rolling a (non biased) die ?
* That I learn something new during this talk ?

Do these two questions refer to the exact same meaning of the concept of *probability* ?

Probability axioms
========================================================
incremental: true
type: lineheight

Une probabilité est une valeur numérique assignée à un événement ($A$), compris comme une possibilité appartenant à l'univers $\Omega$ (l'ensemble de toutes les issues possibles). Les probabilités se conforment aux axiomes suivants (pour des événements mutuellement exclusifs):

+ **Non-negativity.** $P(A_{i}) \geq 0$
+ **Normalisation.** $P(\Omega)=1$
+ **Additivity.** $P(A_{1}\cup A_{2}\cup \ldots)= \sum_{i=1}^{+\infty}P(A_{i})$

<small>*Kolmogorov, A.N. (1933). "Foundations of the Theory of Probability”.*</small>

Interprétation classique (ou théorique)
========================================================
incremental: true
type: lineheight

$$P(pair)=\frac{nombre\ de\ cas\ favorables}{nombre\ de\ cas\ possibles}=\frac{3}{6}=\frac{1}{2}$$

Problème: cette définition est uniquement applicable aux situations dans lesquelles il n'y a qu'un nombre «fini» de résultats possibles équiprobables...

> Quelle est la probabilité qu'il pleuve demain ?

> $$P(pluie)=\frac{pluie}{ \{ pluie,\ non-pluie \} }=\frac{1}{2}??$$

Interprétation fréquentiste (ou empirique)
========================================================
incremental: true
type: lineheight

$$P(x)=\lim_{n_{t} \to \infty}\frac{n_{x}}{n_{t}}$$

Où $n_{x}$ est le nombre d'occurrences de l'événement $x$ et $n_{t}$ le nombre total d'essais. La conception **fréquentiste** postule que, à long-terme (i.e., quand le nombre d'essais s'approche de l'infini), la fréquence relative va converger *exactement* vers ce qu'on appelle probabilité.

Conséquence: le concept de probabilité s'applique uniquement aux **collectifs**, et non pas aux événéments singuliers.

Interprétation fréquentiste (ou empirique)
========================================================
incremental: false
type: lineheight
    

```r
library(tidyverse)

sample(c(0, 1), 500, replace = TRUE) %>%
        data.frame %>%
        mutate(x = seq_along(.), y = cumsum(.) / seq_along(.) ) %>%
        ggplot(aes(x = x, y = y), log = "y") +
        geom_line(lwd = 1) +
        geom_hline(yintercept = 0.5, lty = 3) +
        xlab("nombre de lancers") +
        ylab("proportion de faces") +
        ylim(0, 1) +
        theme_bw(base_size = 20)
```

<img src="day1_intro_bayes-figure/unnamed-chunk-1-1.png" title="plot of chunk unnamed-chunk-1" alt="plot of chunk unnamed-chunk-1" style="display: block; margin: auto;" />

Objections...
========================================================
incremental: true
type: lineheight

+ Quelle classe de référence ? *Quelle est la probabilité que je vive jusqu'à 80 ans ?*

- Quid des événements qui ne peuvent pas se répéter ? *Quelle est la probabilité que j'apprenne quelque chose pendant cete formation ?*

- A partir de combien de lancers (d'une pièce par exemple) a-t-on une bonne approximation de la probabilité ? Une classe finie d'événements de taille $n$ ne peut produire que des fréquences relatives à une certaine "résolution", $1/n$.

Interprétation propensionniste
========================================================
incremental: true
type: lineheight

Les propriétés fréquentistes (i.e., à long terme) des objets (e.g., une pièce) seraient provoquées par des propriétés physiques intrinsèques aux objets. Par exemple, une pièce biaisée va engendrer une fréquence relative (et donc une probabilité) biaisée en raison de ses propriétés physiques. Pour les propensionnistes, les probabilités représentent ces caractéristiques intrinsèques, ces **propensions** à générer certaines fréquences relatives, et non les fréquences relatives en elles-mêmes.

Conséquence: ces propriétés sont les propriétés d'événements individuels... et non de séquences ! L'interpétation propensionniste nous permet donc de parler de la probabilité d'événements uniques.

Interprétation logique
========================================================
incremental: true
type: lineheight

<center>Il y a 10 étudiants dans cette salle</center>
<center>9 portent un t-shirt <font color = "green">vert</font></center>
<center>1 porte un t-shirt <font color = "red">rouge</font></center>
<center>Une personne est tirée au sort...</center>

<hr width = "75%%" size = "2" align = "center" noshade>

> <center>Conclusion 1: l'étudiant tiré au sort porte un t-shirt &#10004; </center>

<hr width = "75%%" size = "1" align = "center" noshade>

> <center>Conclusion 2: l'étudiant tiré au sort porte un t-shirt <font color = "green">vert</font> &#10007; </center>

<hr width = "75%%" size = "1" align = "center" noshade>

> <center>Conclusion 3: l'étudiant tiré au sort porte un t-shirt <font color = "red">rouge</font> &#10007; </center>

Interprétation logique
========================================================
incremental: false
type: lineheight

L'interprétation logique du concept de probabilité essaye de généraliser la logique (vrai / faux) au monde probabiliste. La probabilité représente donc le *degré de support logique* qu'une conclusion peut avoir, relativement à un ensemble de prémices (Keynes, 1921; Carnap, 1950).

Conséquence: toute probabilité est **conditionnelle**.

Interprétation bayésienne
========================================================
incremental: true
type: lineheight

**La probabilité est une mesure du degré d'incertitude**. Un événement *certain* aura donc une probabilité de 1 et un événement *impossible* aura une probabilité de 0.

Pour parler de probabilités, dans ce cadre, nous n'avons donc plus besoin de nous référer à la limite d'occurrence d'un événement (fréquence).

> "So to assign equal probabilities to two events is not in any way an assertion that they must occur equally often in any random experiment [...], it is only a formal way of saying I don’t know." - Jaynes (1986)

Interprétations probabilistes
========================================================
type: lineheight

+ Interprétation classique ou théorique (Laplace, Bernouilli, Leibniz)
+ **Interprétation fréquentiste ou empirique** (Venn, Reichenbach, von Mises)
+ Interprétation propensionniste (Popper, Miller)
+ Interprétation logique (Keynes, Carnap)
+ **Interprétation subjective ou bayésienne** (Jeffreys, de Finetti, Savage)

*[Plus de détails ici](http://plato.stanford.edu/entries/probability-interpret/)*...

Interprétations probabilistes - résumé
========================================================
incremental: true
type: lineheight

> **Probabilité epistémique (subjective)**
- toute probabilité est conditionnelle à de l'information disponible (e.g., prémices ou données)
- la probabilité comme moyen de quantifier l'incertitude
- probabilité logique, probabilité bayésienne

<hr style="height:20pt; visibility:hidden;" />

> **Probabilité physique (objective)**
- les probabilités dépendent d'un état du monde, de caractéristiques physiques, elles sont indépendantes de l'information disponible (ou de l'incertitude)
- probabilité classique, probabilité fréquentiste </center>

========================================================
type: black

<img src="pill.jpg" height="1000px" width="1600px" />

Aparté - Vous avez dit "hasard" ?
========================================================
type: lineheight
incremental: true

Une suite de nombres au hasard...


```
[1] 79 27 98 28 42
```

`runif, rnorm, rbinom...`


```r
RNGkind()[1] # default random number generator
```

```
[1] "Mersenne-Twister"
```

Mersenne-Twister
========================================================
type: lineheight
incremental: false

Inventé par Makoto Matsumoto et Takuji Nishimura en 1997, le [Mersenne Twister](https://en.wikipedia.org/wiki/Mersenne_Twister) est un générateur de nombres pseudo-aléatoires (PRNG). L'algorithme Mersenne-Twister est le PRNG par défaut en Python, Ruby, R, PHP, Matlab...

<div align = "center">
<img src = "mersenne.png" width = 1200 height = 600>
</div>

Hasard déterministe
========================================================
type: lineheight
incremental: true

$$ Y_{i} \sim \mathrm{Uniform}(10,100) $$


```r
set.seed(666)
as.integer(runif(5, 10, 100) )
```

```
[1] 79 27 98 28 42
```

Si je connais la *graine* d'origine (i.e., le "départ" de l'algorithme) et le fonctionnement précis de l'algorithme, je peux prédire les nombres qui vont être générés. Il sera alors difficile de soutenir que ces nombres auront été générés au "hasard"...


```
[1] 79 27 98 28 42 76
```


```
[1] 79 27 98 28 42 76 98
```

Le langage de l'incertain
========================================================
type: lineheight
incremental: true

"Le hasard est un concept indiquant l'aspect aléatoire des choses, des événements ou encore des décisions, signalant ainsi l'impossibilité de prévoir avec certitude un fait quelconque. Ainsi, pour éclairer le sens du mot, il peut-être dit que le hasard est synonyme d'« imprévisibilité », ou « imprédictibilité »" ([Wikipédia](https://fr.wikipedia.org/wiki/Hasard)).

> "Ce que nous appelons hasard n’est et ne peut être que la cause ignorée d’un effet connu." -- Voltaire

> "Le hasard, ce sont les lois que nous ne connaissons pas." -- Émile Borel

Le hasard apparait donc comme un **état subjectif d'indétermination des causes**. Pour parler de l'incertain (pour quantifier l'incertitude), on utilise les probabilités comme langage.

> "Randomness is a proxy for lack of knowledge" -- Richard McElreath

Un peu de logique
========================================================
type: lineheight
incremental: true

<div align = "center" style="float: bottom;">
<img src = "penguin.jpg" width = 600 height = 600>
</div>

Un peu de logique
========================================================
type: lineheight
incremental: true

**Example 1**

- Si un suspect ment, il transpire.
- (On observe que) ce suspect transpire.
- Par conséquent, ce suspect ment.

**Example 2**

- Si le suspect ment, il va présenter une réaction galvanique cutanée (parce qu'il va se mettre à transpirer).
- (On observe que) Le suspect ne présente pas de réaction galvanique cutanée.
- Par conséquent, le suspect ne ment pas.

**Example 3**

- Tous les menteurs transpirent.
- (On observe que) Ce suspect ne transpire pas.
- Par conséquent, ce suspect n'est pas un menteur.

Inférences invalides
========================================================
type: lineheight
incremental: true

- *affirmation du conséquent*: $\dfrac{A \Rightarrow B, \ B}{A}$

- Si il a plu, alors le sol est mouillé (A implique B). Le sol est mouillé (B). Donc il a plu (A).

- *négation de l'antécédent*: $\dfrac{A \Rightarrow B, \ \neg A}{\neg B}$

- Si il a plu, alors le sol est mouillé (A implique B). Il n'a pas plus (non A). Donc le sol n'est pas mouillé (non B).

Inférences valides
========================================================
type: lineheight
incremental: true

- *modus ponens*: $\dfrac{A \Rightarrow B, \ A}{B}$

- Si on est lundi, alors John ira au travail (A implique B). On est lundi (A). Donc John ira au travail (B).

- *modus tollens*: $\dfrac{A \Rightarrow B, \ \neg B}{\neg A}$

- Si mon chien détecte un intru, alors il aboie (A implique B). Mon chien n'a pas aboyé (non B). Donc il n'a pas détecté d'intrus (non A).

Logique, fréquentisme et raisonnement probabiliste
========================================================
incremental: true
type: lineheight

Le *modus tollens* est un des raisonnements logiques les plus importants et les plus performants. Dans le cadre de l'inférence statistique, il s'applique parfaitement au cas suivant: "Si $H_{0}$ est vraie, alors $x$ ne devrait pas se produire. On observe $x$. Alors $H_{0}$ est fausse".

Mais nous avons le plus souvent affaire à des hypothèses "continues", probabilistes.

L'inférence fréquentiste (fishérienne) est elle aussi probabiliste, de la forme "Si $H_{0}$ est vraie, alors $x$ est peu probable. On observe $x$. Alors $H_{0}$ est peu probable."

Or cet argument est invalide, le *modus tollens* ne s'applique pas au monde probabiliste (e.g., Pollard & Richardson, 1987; Rouder, Morey, Verhagen, Province, & Wagenmakers, 2016). 

Par exemple: *si un individu est un homme, alors il est peu probable qu'il soit pape. Francois est pape. François n'est donc certainement pas un homme...*

L'échec de la falsification
========================================================
type: lineheight
incremental: true

Forme logique d'un test expérimental d'une théorie $T$ (from Meehl, 1990)

$$(T \land A_{t} \land C_{p} \land A_{i} \land C_{n}) \to (O1 \supset O2)$$

Où $A_{t}$ est une conjonction d'hypothèses auxiliaires, $C_{p}$ est le *ceteribus paribus* ("all other things being equal"), $A_{i}$ est une théorie concernant les instruments, et $C_{n}$ un énoncé à propos des conditions particulières de cette expérience (*particulars*).

Observer $(O1 \land \lnot O2)$ n'est pas suffisant pour falsifier $T$, car cela revient à dire que toute la partie gauche est fausse. Or, dire qu'une conjonction est fausse revient à énoncer une disjunction des conjoints (i.e., $(T \lor A_{t} \lor C_{p} \lor A_{i} \lor C_{n})$).

En résumé, on a falsifié une combinaison de tous les élements de la partie gauche, mais observer $(O1 \land \lnot O2)$, ne falsifie pas (le coeur) de la théorie $T$ (see also Lakatos, 1970).

L'échec de la falsification
========================================================
type: lineheight
incremental: true

*Poppérisme naïf*: la science progresse par falsification logique, donc la statistique devrait viser la falsification. Mais...

- les hypothèses (théoriques) ne sont pas les modèles (hypothèses statistiques)

> "Models are devices that connect theories to data. A model is an instanciation of a theory as a set of probabilistic statements" -- (Rouder et al., 2016)

- nos hypothèses sont souvent probabilistes
- erreur de mesure
- la falsification concerne le problème de la démarcation, pas celui de la méthode
- la science est une technologie sociale, la falsification est **consensuelle**, et non pas logique

Notre stratégie
========================================================
type: lineheight
incremental: true

Besoin d'un cadre pour développer des modèles cohérents. Nos outils:

- *Bayesian data analysis*, utiliser les probabilités pour décrire l'incertitude, et étendre la logique au monde probabiliste
- *multilevel modeling*, des modèles à multiples niveaux d'incertitude
- approche par comparaison de modèle, au lieu d'essayer de falsifier un *null model*, comparer des modèles intéressants (AIC, WAIC)

Comparaison de modèles
========================================================
type: lineheight
incremental: false

On s'intéresse au lien entre deux variables aléatoires continues, $x$ et $y$.

<img src="day1_intro_bayes-figure/unnamed-chunk-7-1.png" title="plot of chunk unnamed-chunk-7" alt="plot of chunk unnamed-chunk-7" style="display: block; margin: auto;" />

Comparaison de modèles
========================================================
type: lineheight
incremental: false

L'hypothèse de modélisation la plus simple est de postuler une relation linéaire.

<img src="day1_intro_bayes-figure/unnamed-chunk-8-1.png" title="plot of chunk unnamed-chunk-8" alt="plot of chunk unnamed-chunk-8" style="display: block; margin: auto;" />

Comparaison de modèles
========================================================
type: lineheight
incremental: false

Cette description peut-être *améliorée* pour mieux prendre en compte les données qui s'écartent de la prédiction linéaire.

<img src="day1_intro_bayes-figure/unnamed-chunk-9-1.png" title="plot of chunk unnamed-chunk-9" alt="plot of chunk unnamed-chunk-9" style="display: block; margin: auto;" />

Comparaison de modèles
========================================================
type: lineheight
incremental: true

Un ensemble de $N$ points peut être *exhaustivement* (sans erreur) décrit par une fonction polynomiale d'ordre $N-1$. Augmenter la complexité du modèle améliore donc la précision de notre description des données mais réduit la généralisabilité de ses prédictions (*bias-variance tradeoff*).

<img src="day1_intro_bayes-figure/unnamed-chunk-10-1.png" title="plot of chunk unnamed-chunk-10" alt="plot of chunk unnamed-chunk-10" style="display: block; margin: auto;" />

Nous avons besoin d'outils qui prennent en compte le rapport *qualité de la description* / *complexité*, c'est à dire la **parcimonie** des modèles (AIC, WAIC).
 
Inférence bayésienne
========================================================
type: lineheight
incremental: true

Dans ce cadre, pour chaque problème, nous allons suivre 3 étapes:

- Construire le modèle (l'histoire des données): *likelihood* + *prior*
- Mettre à jour grâce aux données (*updating*), calculer la probabilité *a posteriori*
- Evaluer le modèle, *fit*, sensibilité, résumer les résultats, ré-ajuster

> "Bayesian inference is really just counting and comparing of possibilities [...] in order to make good inference about what actually happened, it helps to consider everything that could have happened." -- McElreath (2015).

Exercice - Problème du sac de billes
========================================================
type: lineheight
incremental: true

Imaginons que nous disposions d'un sac, contenant 4 billes. Ces billes peuvent être soit blanches, soit bleues. Nous savons qu'il y a précisémment 4 billes, mais nous ne connaissons pas le nombre de billes blanches, ou bleues.

Nous savons cependant qu'il existe cinq possibilités (que nous considérerons comme nos *hypothèses*):

> <p align = "center"> &#9898 &#9898 &#9898 &#9898</p>
> <p align = "center"> &#128309 &#9898 &#9898 &#9898</p>
> <p align = "center">&#128309 &#128309 &#9898 &#9898</p>
> <p align = "center">&#128309 &#128309 &#128309 &#9898</p>
> <p align = "center">&#128309 &#128309 &#128309 &#128309</p>

Exercice - Problème du sac de billes
========================================================
type: lineheight
incremental: true

Le but est de déterminer quelle combinaison serait la plus probable, **sachant certaines observations**. Imaginons que l'on tire trois billes à la suite, avec remise, et que l'on obtienne la séquence suivante:

<p align = "center">&#128309 &#9898 &#128309</p>

Cette séquence représente nos données. A partir de là, quelle **inférence** peut-on faire sur le contenu du sac ? En d'autres termes, que peut-on dire de la probabilité de chaque hypothèse ?

> <p align = "center"> &#9898 &#9898 &#9898 &#9898</p>
> <p align = "center"> &#128309 &#9898 &#9898 &#9898</p>
> <p align = "center">&#128309 &#128309 &#9898 &#9898</p>
> <p align = "center">&#128309 &#128309 &#128309 &#9898</p>
> <p align = "center">&#128309 &#128309 &#128309 &#128309</p>

Enumérer les possibilités
========================================================
type: lineheight
incremental: false

<p align = "center"> hypothèse: &#128309 &#9898 &#9898 &#9898 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; données: &#128309 </p>

<img src="day1_intro_bayes-figure/unnamed-chunk-11-1.png" title="plot of chunk unnamed-chunk-11" alt="plot of chunk unnamed-chunk-11" style="display: block; margin: auto;" />

Enumérer les possibilités
========================================================
type: lineheight
incremental: false

<p align = "center"> hypothèse: &#128309 &#9898 &#9898 &#9898 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; données: &#128309 &#9898</p>

<img src="day1_intro_bayes-figure/unnamed-chunk-12-1.png" title="plot of chunk unnamed-chunk-12" alt="plot of chunk unnamed-chunk-12" style="display: block; margin: auto;" />

Enumérer les possibilités
========================================================
type: lineheight
incremental: false

<p align = "center"> hypothèse: &#128309 &#9898 &#9898 &#9898 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; données: &#128309 &#9898 &#128309</p>

<img src="day1_intro_bayes-figure/unnamed-chunk-13-1.png" title="plot of chunk unnamed-chunk-13" alt="plot of chunk unnamed-chunk-13" style="display: block; margin: auto;" />

Enumérer les possibilités
========================================================
type: lineheight
incremental: false

<p align = "center"> hypothèse: &#128309 &#9898 &#9898 &#9898 &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; données: &#128309 &#9898 &#128309 </p>

<img src="day1_intro_bayes-figure/unnamed-chunk-14-1.png" title="plot of chunk unnamed-chunk-14" alt="plot of chunk unnamed-chunk-14" style="display: block; margin: auto;" />

Enumérer les possibilités
========================================================
incremental: false
type: lineheight

Sous cette hypothèse, $3$ chemins (sur $4^{3}=64$) conduisent au résultat obtenu. Qu'en est-il des autres hypothèses?

<p align = "center"> &#9898 &#9898 &#9898 &#128309 &emsp;&emsp;&emsp;&emsp; &#9898 &#128309 &#128309 &#128309 &emsp;&emsp;&emsp;&emsp; &#9898 &#9898 &#128309 &#128309 </p>

<img src="day1_intro_bayes-figure/unnamed-chunk-15-1.png" title="plot of chunk unnamed-chunk-15" alt="plot of chunk unnamed-chunk-15" style="display: block; margin: auto;" />

Comparer les hypothèses
========================================================
incremental: false
type: lineheight

<p align = "center"> &#128309 &#128309 &#128309 &#9898 </p>

Au vu des données, cette hypothèse est la plus *probable* car c'est l'hypothèse qui maximise le nombre de manières possibles d'obtenir les données obtenues.

<center>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:30px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg th{font-family:Arial, sans-serif;font-size:30px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg .tg-gq9a{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center}
.tg .tg-k9ij{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-gq9a">Hypothèse</th>
    <th class="tg-gq9a">Façons d'obtenir les données</th>
  </tr>
  <tr>
    <td class="tg-gq9a"> <p> &#9898 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a"> 0 x 4 x 0 = 0</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">1 x 3 x 1 = 3</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">2 x 2 x 2 = 8 </td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#128309 &#9898 </p></td>
    <td class="tg-gq9a">3 x 1 x 3 = 9</td>
  </tr>
  <tr>
    <td class="tg-k9ij"><p> &#128309 &#128309 &#128309 &#128309 </p></td>
    <td class="tg-k9ij">4 x 0 x 4 = 0</td>
  </tr>
</table>

</center>

Accumulation d'évidence
========================================================
incremental: true
type: lineheight

Dans le cas précédent, nous avons considéré que toutes les hypothèses étaient équiprobables a priori (*principe d'indifférence*). Cependant, on pourrait avoir de l'information a priori, provenant de nos connaissances (des particularités des sacs de billes par exemple) ou de données antérieures.

Imaginons que nous tirions une nouvelle bille du sac, comment incorporer cette nouvelle donnée ?

Accumulation d'évidence
========================================================
incremental: false
type: lineheight

Il suffit d'appliquer la même stratégie que précédemment, et de mettre à jour le dernier compte en le multipliant par ces nouvelles données. *Yesterday's posterior is today's prior* (Lindley, 2000).

<center>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:30px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg th{font-family:Arial, sans-serif;font-size:30px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-gq9a{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center}
.tg .tg-k9ij{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-gq9a">Hypothèse</th>
    <th class="tg-gq9a"><p>Façons de produire &#128309</p></th>
    <th class="tg-baqh">Compte précédent</th>
    <th class="tg-baqh">Nouveau compte</th>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#9898 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0 x 0 = 0</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">1</td>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">3 x 1 = 3</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">2</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">8 x 2 = 16</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#128309 &#9898 </p></td>
    <td class="tg-gq9a">3</td>
    <td class="tg-baqh">9</td>
    <td class="tg-baqh">9 x 3 = 27</td>
  </tr>
  <tr>
    <td class="tg-k9ij"><p> &#128309 &#128309 &#128309 &#128309 </p></td>
    <td class="tg-k9ij">4</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0 x 4 = 0</td>
  </tr>
</table>

</center>

Incorporer un prior
========================================================
incremental: false
type: lineheight

Supposons maintenant qu'un employé de l'usine de fabrication des billes nous dise que les billes bleues sont rares... Cet employé nous dit que pour chaque sac contenant 3 billes bleues, ils fabriquent deux sacs en contenant seulement deux, et trois sacs en contenant seulement une. Il nous apprend également que tous les sacs contiennent au moins une bille bleue et une bille blanche.

<center>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:30px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg th{font-family:Arial, sans-serif;font-size:30px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-gq9a{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center}
.tg .tg-k9ij{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-gq9a">Hypothèse</th>
    <th class="tg-gq9a"><p>Compte précédent</p></th>
    <th class="tg-baqh">Prior usine</th>
    <th class="tg-baqh">Nouveau compte</th>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#9898 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0 x 0 = 0</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#9898 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">3</td>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">3 x 3 = 9</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#9898 &#9898 </p></td>
    <td class="tg-gq9a">16</td>
    <td class="tg-baqh">2</td>
    <td class="tg-baqh">16 x 2 = 32</td>
  </tr>
  <tr>
    <td class="tg-gq9a"><p> &#128309 &#128309 &#128309 &#9898 </p></td>
    <td class="tg-gq9a">27</td>
    <td class="tg-baqh">1</td>
    <td class="tg-baqh">27 x 1 = 27</td>
  </tr>
  <tr>
    <td class="tg-k9ij"><p> &#128309 &#128309 &#128309 &#128309 </p></td>
    <td class="tg-k9ij">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0 x 0 = 0</td>
  </tr>
</table>

</center>

Des énumérations aux probabilités
========================================================
incremental: false
type: lineheight

La plausiblité d'une hypothèse après avoir observé certaines données est proportionnelle au nombre de façons qu'a cette hypothèse de produire les données observées, multiplié par sa plausibilité a priori.

$$p(hypothesis|data)\propto p(data|hypothesis) \times p(hypothesis)$$

Pour passer des *plausibilités* aux *probabilités*, il suffit de standardiser ces plausibilités pour que la somme des plausibilités de toutes les hypothèses possibles soit égale à $1$.

$$p(hypothesis|data) = \frac{p(data|hypothesis) \times p(hypothesis)}{sum\ of\ products}$$

Des énumérations aux probabilités
========================================================
incremental: true
type: lineheight

Définissons $p$ comme la proportion de billes bleues.

<center>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:20px;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg th{font-family:Arial, sans-serif;font-size:20px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:0px;overflow:hidden;word-break:normal;border-top-width:1px;border-bottom-width:1px;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-gq9a{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center}
.tg .tg-k9ij{font-family:"Lucida Sans Unicode", "Lucida Grande", sans-serif !important;;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-gq9a">Hypothèse</th>
    <th class="tg-gq9a">p</th>
    <th class="tg-baqh">Manières de produire les données</th>
    <th class="tg-baqh">Plausibilité</th>
  </tr>
  <tr>
    <td class="tg-gq9a"> &#9898 &#9898 &#9898 &#9898 </td>
    <td class="tg-gq9a">0</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0</td>
  </tr>
  <tr>
    <td class="tg-gq9a"> &#128309 &#9898 &#9898 &#9898 </td>
    <td class="tg-gq9a">0.25</td>
    <td class="tg-baqh">3</td>
    <td class="tg-baqh">0.15</td>
  </tr>
  <tr>
    <td class="tg-gq9a"> &#128309 &#128309 &#9898 &#9898 </td>
    <td class="tg-gq9a">0.5</td>
    <td class="tg-baqh">8</td>
    <td class="tg-baqh">0.40</td>
  </tr>
  <tr>
    <td class="tg-gq9a"> &#128309 &#128309 &#128309 &#9898 </td>
    <td class="tg-gq9a">0.75</td>
    <td class="tg-baqh">9</td>
    <td class="tg-baqh">0.45</td>
  </tr>
  <tr>
    <td class="tg-k9ij"> &#128309 &#128309 &#128309 &#128309 </td>
    <td class="tg-k9ij">1</td>
    <td class="tg-baqh">0</td>
    <td class="tg-baqh">0</td>
  </tr>
</table>

</center>


```r
ways <- c(0, 3, 8, 9, 0)
ways / sum(ways)
```

```
[1] 0.00 0.15 0.40 0.45 0.00
```

Notations, terminologie
========================================================
incremental: false
type: lineheight

- $\theta$ un paramètre ou vecteur de paramètres (e.g., la proportion de billes bleues)
- $\color{orangered}{p(x\vert \theta)}$ <span style="color:orangered"> la distribution de probabilité conditionnelle des données $x$ sachant le paramètre $\theta$. Une fois que la valeur de $x$ est connue, est vue comme la fonction de vraissemblance (*likelihood*) du paramètre $\theta$</span>
- $\color{steelblue}{p(\theta)}$ <span style="color:steelblue"> la distribution de probabilité a priori de $\theta$</span>
- $\color{purple}{p(\theta \vert x)}$ <span style="color:purple"> la distribution de probabilité a posteriori de $\theta$ (sachant $x$)</span>
- $\color{green}{p(x)}$ <span style="color:green"> la distribution de probabilité marginale de $x$ (sur $\theta$)</span>

<br>

$$\color{purple}{p(\theta \vert x)} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{p(x)}} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{\sum\limits_{\theta}p(x|\theta)p(\theta)}} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{\int\limits_{\theta}p(x|\theta)p(\theta)\mathrm{d}x}} \propto \color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}$$

Loi de probabilité, cas discret
========================================================
type: lineheight
incremental: true

Une fonction de masse (*probability mass function*, ou *PMF*) est une fonction qui attribue une probabilité à chaque valeur d'une variable aléatoire. Exemple de la distribution binomiale pour une pièce non biaisée ($\theta=0.5$).

<img src="day1_intro_bayes-figure/unnamed-chunk-17-1.png" title="plot of chunk unnamed-chunk-17" alt="plot of chunk unnamed-chunk-17" style="display: block; margin: auto;" />


```r
dbinom(x = 0:10, size = 10, prob = 0.5) %>% sum
```

```
[1] 1
```

Cas continu
========================================================
type: lineheight
incremental: true

Une densité de probabilité (*probability density function*, ou *PDF*), est une fonction qui permet de représenter une loi de probabilité sous forme d'intégrales.

<img src="day1_intro_bayes-figure/unnamed-chunk-19-1.png" title="plot of chunk unnamed-chunk-19" alt="plot of chunk unnamed-chunk-19" style="display: block; margin: auto;" />


```r
integrate(dnorm, -Inf, Inf, mean = 100, sd = 15)
```

```
1 with absolute error < 1.3e-06
```

Aparté, qu'est-ce qu'une intégrale ?
========================================================
type: lineheight
incremental: true

Une intégrale correspond à la **surface** (aire géométrique) délimitée par la représentation graphique d'une fonction, *l'aire sous la courbe*. Une distribution est dite **impropre** si son intégrale n'est pas égale à un nombre fini (e.g., $+ \infty$), et **normalisée** si son intégrale est égale à 1.

<img src="day1_intro_bayes-figure/unnamed-chunk-21-1.png" title="plot of chunk unnamed-chunk-21" alt="plot of chunk unnamed-chunk-21" style="display: block; margin: auto;" />

Aparté, qu'est-ce qu'une intégrale ?
========================================================
type: lineheight
incremental: true

<img src="day1_intro_bayes-figure/unnamed-chunk-22-1.png" title="plot of chunk unnamed-chunk-22" alt="plot of chunk unnamed-chunk-22" style="display: block; margin: auto;" />

L'intégrale de $f(x)$ sur l'intervalle [90 ; 96] vaut: $\int_{90}^{96} f(x) \ \mathrm{d}x = 0.142$.


```r
integrate(dnorm, 90, 96, mean = 100, sd = 15)
```

```
0.1423704 with absolute error < 1.6e-15
```

Probabilité conjointe
========================================================
incremental: false
type: lineheight


```r
library(magrittr)

data(HairEyeColor) # data adapted from Snee (1974)

cont <- apply(HairEyeColor, c(1, 2), sum) %>% t 
cont <- round(cont / sum(cont), 2)
cont
```

```
       Hair
Eye     Black Brown  Red Blond
  Brown  0.11  0.20 0.04  0.01
  Blue   0.03  0.14 0.03  0.16
  Hazel  0.03  0.09 0.02  0.02
  Green  0.01  0.05 0.02  0.03
```

Dans chaque cellule, on trouve la **probabilité conjointe** d'avoir telle couleur de cheveux **ET** telle couleur d'yeux, qui s'écrit $p(c,y)=p(y,c)$.

Probabilité marginale
========================================================
incremental: false
type: lineheight


```r
cont2 <- cont %>% as.data.frame %>% mutate(marginal_eye = rowSums(cont) )
rownames(cont2) <- row.names(cont)
cont2
```

```
      Black Brown  Red Blond marginal_eye
Brown  0.11  0.20 0.04  0.01         0.36
Blue   0.03  0.14 0.03  0.16         0.36
Hazel  0.03  0.09 0.02  0.02         0.16
Green  0.01  0.05 0.02  0.03         0.11
```

On peut aussi s'intéresser à la probabilité d'avoir des yeux bleus, de manière générale. Il s'agit de la probabilité **marginale** de l'événement *yeux bleus*, qui s'obtient par la somme de toutes les probabilités jointes impliquant l'évenement *yeux bleus*. Elle s'écrit $p(y)=\sum\limits_{c}p(y|c)p(c)$.

Probabilité marginale
========================================================
incremental: false
type: lineheight


```r
cont3 <- rbind(cont2, colSums(cont2) )
rownames(cont3) <- c(row.names(cont2), "marginal_hair")
cont3
```

```
              Black Brown  Red Blond marginal_eye
Brown          0.11  0.20 0.04  0.01         0.36
Blue           0.03  0.14 0.03  0.16         0.36
Hazel          0.03  0.09 0.02  0.02         0.16
Green          0.01  0.05 0.02  0.03         0.11
marginal_hair  0.18  0.48 0.11  0.22         0.99
```

On peut bien entendu aussi s'intéresser aux probabilités des couleurs de cheveux, de manière générale. Elle s'écrit $p(c)=\sum\limits_{y}p(c|y)p(y)$.

Probabilité conditionnelle
========================================================
incremental: true
type: lineheight


```
     Black Brown  Red Blond marginal_eye
Blue  0.03  0.14 0.03  0.16         0.36
```

On pourrait aussi s'intéresser à la probabilité qu'une personne ait les cheveux blonds, **sachant** qu'elle a les yeux bleus. Il s'agit d'une probabilité **conditionnelle**, et s'écrit $p(c|y)$. Cette probabilité conditionnelle peut se ré-écrire: $p(c|y)= \frac{p(c,y)}{p(y)}$.


```r
cont3["Blue","Blond"] / cont3["Blue","marginal_eye"]  
```

```
     Blue 
0.4444444 
```

Probabilité conditionnelle, règle du produit
========================================================
incremental: true
type: lineheight

On remarque dans le cas précédent que $p(blonds|bleus)$ **n'est pas nécessairement égal** à $p(bleus|blonds)$.

Autre exemple: la probabilité de mourir sachant qu'on a été attaqué par un requin n'est pas la même que la probabilité d'avoir été attaqué par un requin, sachant qu'on est mort.

De la même manière, $p(data|H_{0}) \neq p(H_{0}|data)$ -> [*"confusion of the inverse"*](https://en.wikipedia.org/wiki/Confusion_of_the_inverse).

A partir des axiomes de Kolmogorov (cf. début du cours), et des définitions précédentes des probabilités conjointes, marginales, et conditionnelles, découle la **règle du produit**:

$$p(a,b)=p(b)p(a|b)=p(a)p(b|a)$$

Dérivation du théorème de Bayes
========================================================
incremental: true
type: lineheight

$$p(x,y) = p(x|y)p(y) = p(y|x)p(x)$$

$$p(y|x)p(x) = p(x|y)p(y)$$

$$p(y|x) = \dfrac{p(x|y)p(y)}{p(x)}$$

$$p(x|y) = \dfrac{p(y|x)p(x)}{p(y)}$$

$$p(hypothesis|data) = \frac{p(data|hypothesis) \times p(hypothesis)}{sum\ of\ products}$$

Exemple - Diagnostique médical (Gigerenzer, 2002)
========================================================
incremental: true
type: lineheight

- Chez les femmes âgées de 40-50 ans, sans antécédents familiaux et sans symptômes, la probabilité d'avoir un cancer du sein est de .008.

- Propriétés de la mammographie:
  - Si une femme a un cancer du sein, la probabilité d'avoir un résultat positif est de .90
  - Si une femme n'a pas de cancer du sein, la probabilité d'avoir un résultat positif est de .07

- Imaginons qu'une femme passe une mammographie, et que le test est positif. Que doit-on **inférer** ? Quelle est la probabilité que cette femme ait un cancer du sein ?

Logique du Maximum Likelihood
========================================================
incremental: true
type: lineheight

- Une approche générale de l'estimation de paramètre
- Les paramètres **gouvernent** les données, les données **dépendent** des paramètres
  - Sachant certaines valeurs des paramètres, nous pouvons calculer la **probabilité conditionelle** des données observées
  - Le résultat de la mammographie (i.e., les données) dépend de la présence / absence d'un cancer du sein (i.e., le paramètre)
- L'approche par *maximum likelihood* (ML) pose la question: "*Quelles sont les valeurs du paramètre qui rendent les données observées les plus probables ?*"

Maximum Likelihood
========================================================
incremental: true
type: lineheight

- Spécifier la probabilité conditionnelle des données $p(x|\theta)$
- Quand on le considère comme fonction de $\theta$, on parle de **likelihood**: $L(\theta|x)=p(x|\theta)$
- L'approche par ML consiste donc à maximiser cette fonction, en utilisant les valeurs (connues) de $x$

Probabilité conditionnelle
========================================================
incremental: true
type: lineheight

- Si une femme a un cancer du sein, la probabilité d'obtenir un résultat positif est de .90
  - $p(Mam=+|Cancer=+)=.90$ 
  - $p(Mam=-|Cancer=+)=.10$
  
- Si une femme n'a pas de cancer du sein, la probabilité d'obtenir un résultat positif est de .07
  - $p(Mam=+|Cancer=-)=.07$ 
  - $p(Mam=-|Cancer=-)=.93$

Diagnostique médical, maximum likelihood
========================================================
incremental: true
type: lineheight

- Une femme passe une mammographie, le résultat est positif...
  - $p(Mam=+|Cancer=+)=.90$ 
  - $p(Mam=+|Cancer=-)=.07$
  
- ML: quelle est la valeur de *Cancer* qui **maximise** $Mam=+$ ?
  - $p(Mam=+|Cancer=+)=.90$ 

========================================================

<div align = "center" style="border:none;">
<img src = "bayes.png" width = 600 height = 600>
</div>

<center>*Wait a minute...*</center>

Diagnostique médical, fréquences naturelles
========================================================
incremental: true
type: lineheight

- Considérons 1000 femmes âgées de 40 à 50 ans, sans antécédents familiaux et sans symptômes de cancer
  - 8 femmes sur 1000 ont un cancer
- On réalise une mammographie
  - Sur les 8 femmes ayant un cancer, 7 auront un résultat positif
  - Sur les 992 femmes restantes, 69 auront un résultat positif
- Une femme passe une mammographie, le résultat est positif
- Que devrait-on **inférer** ?

Diagnostique médical, fréquences naturelles
========================================================
incremental: true
type: lineheight

<div align = "center" style = "border:none;">
<img src = "diagram.png" width = 850 height = 500>
</div>

<br>

$$p(Cancer=+|Mam=+)=\frac{7}{7+69}=\frac{7}{76}\approx.09$$

Diagnostique médical, théorème de Bayes
========================================================
incremental: true
type: lineheight

$$\color{purple}{p(\theta \vert x)} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{p(x)}}$$

$\color{steelblue}{p(\theta)}$ <span style="color:steelblue"> la probabilité *a priori* de $\theta$: tout ce qu'on sait de $\theta$ avant d'observer les données. Par exemple: $p(Cancer=+)=.008$ et $p(Cancer=-)=.992$.</span>


```r
prior <- c(0.008, 0.992)
```

Diagnostique médical, théorème de Bayes
========================================================
incremental: true
type: lineheight

$$\color{purple}{p(\theta \vert x)} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{p(x)}}$$

$\color{orangered}{p(x\vert \theta)}$ <span style="color:orangered"> probabilité conditionnelle des données ($x$) sachant le paramètre ($\theta$), qu'on appelle aussi la *likelihood* (*ou fonction de vraissemblance*) du paramètre ($\theta$).</span>


```r
like <- rbind(c(0.9, 0.1), c(0.07, 0.93) ) %>% data.frame
colnames(like) <- c("Mam+", "Mam-")
rownames(like) <- c("Cancer+", "Cancer-")
like
```

```
        Mam+ Mam-
Cancer+ 0.90 0.10
Cancer- 0.07 0.93
```

Diagnostique médical, théorème de Bayes
========================================================
incremental: true
type: lineheight

$$\color{purple}{p(\theta \vert x)} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{p(x)}}$$

<span style="color:green"> $p(x)$ la probabilité marginale de $x$ (sur $\theta$). Sert à normaliser la distribution.</span>

$$\color{green}{p(x)=\sum\limits_{\theta}p(x|\theta)p(\theta)}$$


```r
(marginal <- sum(like$"Mam+" * prior) )
```

```
[1] 0.07664
```

Diagnostique médical, théorème de Bayes
========================================================
incremental: true
type: lineheight

$$\color{purple}{p(\theta \vert x)} = \dfrac{\color{orangered}{p(x\vert \theta)} \color{steelblue}{p(\theta)}}{\color{green}{p(x)}}$$

$\color{purple}{p(\theta \vert x)}$ <span style="color:purple"> la probabilité a posteriori de $\theta$ sachant $x$, c'est à dire ce qu'on sait de $\theta$ après avoir pris connaissance de $x$.</span>


```r
(posterior <- (like$"Mam+" * prior ) / marginal )
```

```
[1] 0.09394572 0.90605428
```

L'inférence bayésienne comme mise à jour probabiliste des connaissances
========================================================
incremental: true
type: lineheight

Avant de passer le mammogramme, la probabilité qu'une femme tirée au sort ait un cancer du sein était de $p(Cancer)=.008$ (*prior*). Après un résultat positif, cette probabilité est devenue $p(Cancer|Mam+)=.09$ (*posterior*). Ces probabilités sont des expressions de nos *connaissances*. Après un mammogramme positif, on pense toujours que c'est "très improbable" d'avoir un cancer, mais cette probabilité a considérablement évolué relativement à "avant le test".

> A Bayesianly justifiable analysis is one that *treats known values as observed values of random variables, treats unknown values as unobserved random variables, and calculates the conditional distribution of unknowns given knowns and model specifications using Bayes’ theorem*. -- Rubin (1984)

Monty Hall
========================================================
incremental: true
type: lineheight

<br>

<div align="center">
<video width="1200" height="600" controls="controls">
<source src="montyhall.mp4" type="video/mp4">
</video>
<div>

Monty Hall
========================================================
incremental: false
type: lineheight

<div align = "center" style="border:none;">
<img src = "monty1.png" width = 600 height = 600>
</div>

<center> Que-feriez-vous (intuitivement) ? Analysez ensuite la situation en utilisant le théorème de Bayes. </center>

Monty Hall
========================================================
incremental: true
type: lineheight

Il s'agit d'un problème de probabilités conditionnelles... Définissons les événements suivants:

P1: l'animateur ouvre la porte 1 <br>
P2: l'animateur ouvre la porte 2 <br>
P3: l'animateur ouvre la porte 3 <br>

V1: la voiture se trouve derrière la porte 1 <br>
V2: la voiture se trouve derrière la porte 2 <br>
V3: la voiture se trouve derrière la porte 3 <br>

Si on a choisi la porte n°1 et que l'animateur a choisi la porte n°3 (*et qu'il sait où se trouve la voiture*), il s'ensuit que:

$p(P3|V1)=\dfrac{1}{2}$, $p(P3|V2)=1$, $p(P3|V3)=0$

Monty Hall
========================================================
incremental: true
type: lineheight

On sait que $p(V3|P3)=0$, on veut connaître $p(V1|P3)$ et $p(V2|P3)$ afin de pouvoir choisir. Résolution par le théorème de Bayes.

$p(V1|P3)=\dfrac{p(P3|V1) \times p(V1)}{p(P3)}=\dfrac{\dfrac{1}{2} \times \dfrac{1}{3}}{\dfrac{1}{2}}=\dfrac{1}{3}$

$p(V2|P3)=\dfrac{p(P3|V2) \times p(V2)}{p(P3)}=\dfrac{1 \times \dfrac{1}{3}}{\dfrac{1}{2}}=\dfrac{2}{3}$

Monty Hall
========================================================
incremental: true
type: lineheight

<div align = "center" style="border:none;">
<img src = "monty2.png" width = 600 height = 600>
</div>

Morale
========================================================
incremental: true
type: lineheight

Nos intuitions probabilistes sont, dans la grande majorité des cas, très mauvaises. Au lieu de compter sur elles, il est plus sage de se reposer sur des règles logiques (*modus ponens* et *modus tollens*) et probabilistes simples (règle du produit, règle de la somme, théorème de Bayes), nous assurant de réaliser l'inférence logique la plus juste.

Autrement dit, "Don't be clever" (McElreath, 2015).

Further resources
========================================================
incremental: false
type: lineheight

Books

- McElreath, R. (2015). *Statistical Rethinking: A Bayesian Course with Examples in R and Stan*. CRC Press.
- Kruschke, J. K. (2015). *Doing Bayesian Data Analysis, Second Edition: A Tutorial with R, JAGS, and Stan*. Academic Press / Elsevier.
-  Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., Rubin, D. B. (2013). *Bayesian Data Analysis, third edition*. London: CRC Press.
- Noël, Y. (2013). *Psychologie statistique avec R*. Springer.

Slides are available here: https://github.com/lnalborczyk

References
========================================================
incremental: false
type: lineheight

Carnap, R. (1950). *Logical foundations of probability*. Chicago, IL, US: University of Chicago Press.

Jaynes, E. (1986). Bayesian Methods: General Background. In J. Justice (Author), Maximum Entropy and Bayesian Methods in Applied Statistics Proceedings of the Fourth Maximum Entropy Workshop University of Calgary, 1984 (pp. 1-25). Cambridge: Cambridge University Press.

Keynes, J. M. (1921). *Treatise on Probability*. London: Macmillan & Co.

Kolmogorov, A. N. (1933). *Grundbegriffe der Wahrscheinlichkeitrechnung, Ergebnisse Der Mathematik*. Translated as *Foundations of Probability*. New York: Chelsea Publishing Company, 1950.

Lakatos, I. (1970). Falsification and the methodology of scientific research programmes. In: Lakatos, Musgrave eds., pp. 91–195.

Lindley, D. (2000). The Philosophy of Statistics. Journal of the Royal Statistical Society. *Series D (The Statistician), 49*(3), 293-337.

References
========================================================
incremental: false
type: lineheight

Meehl, P. (1990). Appraising and amending theories: The strategy of Lakatosian defense and two principles that warrant using it. *Psychological Inquiry, 1*, 108-141, 173-180.

Pollard, P., & Richardson, J. T. E. (1987). On the probability of making type I errors. *Psychological Bulletin, 102*(1), 159-163.

Rouder, J. N., Morey, R. D., & Wagenmakers, E.-J. (2016). The Interplay between Subjectivity, Statistical Practice, and Psychological Science. *Collabra, 2*(1), 1-12.

Rouder, J. N., Morey, R. D., Verhagen, J., Province, J. M., Wagenmakers, E. J. (2016). Is There a Free Lunch in Inference? *Topics in Cognitive Science, 8*, 520-47.

<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
<script>

for(i=0;i<$("section").length;i++) {
if(i==0) continue
$("section").eq(i).append("<p style='font-size:xx-large;position:fixed;right:200px;bottom:50px;'>" + i + "</p>")
}

</script>
