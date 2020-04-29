# Homework 5: Probabilistic models

Student Name: Kuan-Lin Liu

Student ID: kll482

## 1. Logistic Regression

### 1.1 Equivalence of ERM and probabilistic approaches

(a)

$ERM = argmin_w \frac{1}{n}\sum_{i=1}^n log[1+exp(-y_iw^Tx_i)]$

$\hat{R}(w) = \frac{1}{n}\sum_{i=1}^n log[1+exp(-y_iw^Tx_i)]$

(b)

Let $$P(y=1|x;w) = f(w^Tx) = \frac{1}{1+exp(-w^Tx)}$$

We know $$\enspace P(Y=y|X=x)=f(w^Tx)^y[1-f(w^Tx)]^{(1-y)}$$

Then,
$$
\begin{equation}
\begin{cases}
L(w) = \prod_{i=1}^n P(Y=y_i|X=x_i) = \prod_{i=1}^n f(w^Tx)^y[1-f(w^Tx)]^{(1-y)} \\
LL(w) = \log\left[ \prod_{i=1}^n f(w^Tx)^y[1-f(w^Tx)]^{(1-y)} \right] = \sum_{i=1}^n y_i\log f(w^Tx_i) + (1-y_i)\log (1-f(w^Tx_i)) \\
NLL(w) = -\sum_{i=1}^n y_i\log f(w^Tx_i) + (1-y_i)\log (1-f(w^Tx_i))
\end{cases}
\end{equation}
$$

(c)

Prove (a) and (b) are equal

When $\enspace y_i=1 \enspace\text{and} \enspace \hat{y_i} = 1,$
$$NLL(w) = \sum_{i=1}^n-\log f(w^Tx_i) = \sum_{i=1}^n\log (1+exp(-w^Tx_i)) = n\hat{R(w)}$$

When $\enspace y_i=-1 \enspace\text{and} \enspace \hat{y_i} = 0,$

$$NLL(w) = \sum_{i=1}^n-\log (1-f(w^Tx_i))$$

$$= \sum_{i=1}^n \log (1-\frac{1}{1+exp(-w^Tx_i)})^{-1}$$

$$= \sum_{i=1}^n \log (\frac{exp(-w^Tx_i)}{1+exp(-w^Tx_i)})^{-1} $$

$$= \sum_{i=1}^n \log (\frac{1+exp(-w^Tx_i)}{exp(-w^Tx_i)})$$

$$= \sum_{i=1}^n \log (1+exp(w^Tx_i))$$

$$= n\hat{R(w)}$$

Since n is a constant, ERM and MLE will not be affected by the constant and will produce the same w.

### 1.2 Linearly Separable Data

#### 1.2.1

#### 1.2.2

$\frac{\partial NLL(w; c)}{\partial c} = \frac{\partial -\sum_{i=1}^n y_i\log f(cw^Tx_i) + (1-y_i)\log (1-f(cw^Tx_i))}{\partial c}$

Let $z_i = cw^Tx_i$ and $f_i=f(cw^Tx_i)$,

$\frac{\partial NLL(w; c)}{\partial c}=\frac{\partial NLL}{\partial f_i} \cdot \frac{\partial f_i}{\partial z_i} \cdot \frac{\partial z_i}{\partial c}$

$=-\sum_{i=1}^n [\frac{y_i}{f_i} - \frac{1-y}{1-f}]\cdot f_i(1-f_i) \cdot w^Tx_i$

$=-\sum_{i=1}^n[(1-f_i)y_i-f_i(1-y_i)] \cdot w^Tx_i$

$=-\sum_{i=1}^n [y_i-f_i] \cdot w^Tx_i$

$=\sum_{i=1}^n [f_i-y_i] \cdot w^Tx_i$

$=\sum_{i=1}^n [f_i(cw^Tx)-y_i] \cdot w^Tx_i$

$\because \text{If} \ c \rightarrow \infty, \ f(cw^Tx_i) \rightarrow 1$

$\therefore$ the derivative of NLL on c is strictly positive

### 2.1

$P(w|\mathcal{D}) = \frac{P(w\cap\mathcal{D})}{P(\mathcal{D}}=\frac{P(w)\cdot P(\mathcal{D}|w)}{P(\mathcal{D})} $

$\propto P(w)\cdot P(\mathcal{D}|w)$

$\propto P(w)\cdot L(w)$

$\propto P(w)\cdot LL(w)$ 

$ \propto P(w)\cdot -NLL(w)$

### 2.3
Given, $$\mathcal{N}(0, \Sigma)=\frac{1}{\sqrt{2\pi\sigma^2}}exp(-\frac{w^Tw}{2\Sigma})$$

Solving,

$$-\log P(w|\mathcal{D}) \propto -\log[P(w)\cdot exp(-NLL(w))]$$

$$\propto -\log P(w) + NLL(w)$$

$$\propto -[\log (2\pi\Sigma)^{-\frac{1}{2}} - \frac{1}{2}w^Tw\Sigma^{-1}] + n\hat{R(w)}$$

$$\propto \frac{1}{2}log (2\pi\Sigma) + \frac{1}{2}w^Tw\Sigma^{-1} + n\hat{R(w)}$$

$$\propto \frac{1}{2}w^Tw\Sigma^{-1} + n\hat{R(w)} \enspace \because \frac{1}{2}log (2\pi\Sigma) \enspace \text{is constant} $$

Let, $$\Sigma = \frac{1}{2n\lambda}I$$

Then we can get, $$-\log P(w|\mathcal{D}) \propto n\hat{R(w)} + n\lambda ||w||^2$$

### 3.1

$P(x=H|\theta_1, \theta_2) = \Sigma_{\acute{z} \in \{H, T\}}P(x=H, z=\acute{z} \ |\theta_1\theta_2)$

Solve,

$P(x=H, z=\acute{z} \ |\theta_1\theta_2)$

$= \frac{P(x=H, z=\acute{z},\theta_1\theta_2)}{P(\theta_1\theta_2)}$

$=\frac{P(x=H | z=\acute{z}, \  \theta_1\theta_2) \cdot P(z=\acute{z},\ \theta_1\theta_2)}{P(\theta_1 \theta_2)}$

$=P(x=H | z=\acute{z}, \  \theta_2) \cdot P(z=\acute{z}\ | \theta_1\theta_2) \enspace \because \theta_1 \ \text{is independent to} \ x \ \text{given by} \ z, \ \theta_2$

$=P(x=H | z=\acute{z}, \  \theta_2) \cdot P(z=\acute{z}\ | \theta_1) \enspace \because \theta_2 \ \text{is independent to z given by } \theta_1$

Since $P(x=T|z=T)=1$, we only care about the condition of $\acute{z}=H$.

Therefore, $$P(x=H|\theta_1, \theta_2)=P(x=H|z=H, \theta_2)\cdot P(z=H|\theta_1)=\theta_1\theta_2$$

### 3.2

$P(x|\theta_1, \theta_2)=P(x=H|\theta_1, \theta_2) \cdot P(x=T|\theta_1, \theta_2)$

$P(\mathcal{D}|\theta_1, \theta_2)=(\frac{N_r}{n_h+n_t})(\theta_1\theta_2)^{n_h}(1-\theta_1\theta_2)^{n_t}$

$-\log P(\mathcal{D}|\theta_1, \theta_2) \propto -n_h \log \theta_1\theta_2 - n_t \log (1-\theta_1\theta_2)$

By MLE, we want to minimize $-\log P(\mathcal{D}|\theta_1, \theta_2)$

(a) 

$\frac{\partial -\log P(\mathcal{D}|\theta_1, \theta_2)}{\partial \theta_1}=\frac{-n_h}{\theta_1}+\frac{n_t\theta_2}{1-\theta_1\theta_2}=0$

(b)

$\frac{\partial -\log P(\mathcal{D}|\theta_1, \theta_2)}{\partial \theta_2}=\frac{-n_h}{\theta_2}+\frac{n_t\theta_1}{1-\theta_1\theta_2}=0$

$\Rightarrow \frac{n_t\theta_1\theta_2}{n_h}=1-\theta_1\theta_2$ 

Substitute $\frac{n_t\theta_1\theta_2}{n_h}$ with $1-\theta_1\theta_2$ in (a), we will get $\frac{-n_h}{\theta_1}+\frac{n_hn_t}{n_t\theta_1}=0$.

So, $\theta_1, \theta_2$ can not be estimated using MLE.

### 3.3

$P(\mathcal{D_r, D_c}|\theta_1, \theta_2)=P(D_c|\theta_1)P(D_r|\theta_1, \theta_2)$

$-\log P(\mathcal{D_r, D_c}|\theta_1, \theta_2) = -\log P(D_c|\theta_1)P(D_r|\theta_1, \theta_2)$

$\propto \big[-c_h \log \theta_1 - c_t \log (1-\theta_1)\big]+ \big[-n_h \log \theta_1\theta_2 - n_t \log (1-\theta_1\theta_2)\big]$

By MLE, we want to minimize $\propto \big[-c_h \log \theta_1 - c_t \log (1-\theta_1)\big]+ \big[-n_h \log \theta_1\theta_2 - n_t \log (1-\theta_1\theta_2)\big]$

(a)

$\frac{\partial -\log P(\mathcal{D_r, D_c}|\theta_1, \theta_2)}{\partial \theta_1}=-\big[\frac{c_h}{\theta_1}-\frac{c_t}{1-\theta_1} \big]-\big[\frac{n_h}{\theta_1}-\frac{\theta_2n_t}{1-\theta_1\theta_2} \big]=0$

(b)

$\frac{\partial -\log P(\mathcal{D_r, D_c}|\theta_1, \theta_2)}{\partial \theta_2}=-\big[\frac{n_h}{\theta_2}-\frac{n_t\theta_1}{1-\theta_1\theta_2}\big]=0$

$\Rightarrow 1-\theta_1\theta_2=\frac{n_t\theta_1\theta_2}{n_h}$

Substitute $1-\theta_1\theta_2$ with $\frac{n_t\theta_1\theta_2}{n_h}$ in (a), we will get

$\theta_1=\frac{c_h}{c_h+c_t}$

$\theta_2=\frac{n_h}{(n_h+n_t)\theta_1}=\frac{n_h(c_h+c_t)}{(n_h+n_t)c_h}$

### 3.4
Given $g(\theta_1)=\theta_1^{h-1}(1-\theta_1)^{t-1}$

$\theta_{1, MAP}=argmax_{\theta_1}g(\theta_1)L(\theta_1, \theta_2)$

Let $\acute{L}(\theta_1, \theta_2)=g(\theta_1)L(\theta_1, \theta_2)$

$\acute{LL}(\theta_1, \theta_2)=\big[(h-1)\log \theta_1+(t-1) \log (1-\theta_1) \big]+ \big[c_h\log \theta_1+c_t \log(1-\theta_1) \big]+ \big[n_n \log (\theta_1, \theta_2)+n_t \log(1-\theta_1, \theta_2)\big]$

(a)

$\frac{\partial \acute{NLL}}{\partial \theta_1}=-\big[\frac{h-1}{\theta_1}-\frac{t-1}{1-\theta_1} \big]-\big[\frac{n_h}{\theta_1}-\frac{\theta_2n_t}{1-\theta_1\theta_2} \big]-\big[\frac{c_h}{\theta_1}-\frac{c_t}{1-\theta_1} \big]=0$

(b)

$\frac{\partial \acute{NLL}}{\partial \theta_2}=\frac{n_h}{\theta_2}-\frac{\theta_1n_t}{1-\theta_1\theta_2}=0$

$\Rightarrow \frac{n_h}{\theta_1n_t}=\frac{\theta_2}{1-\theta_1\theta_2}$

From (a) and (b), we can obtain

$\theta_1=\frac{c_h+h-1}{c_h+c_t+h+t-2}$

$\theta_2=\frac{n_h}{(n_h+n_t)\theta_1}=\frac{n_h(c_h+c_t+h+t-2)}{(n_h+n_t)(c_h+h-1)}$
