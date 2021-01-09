# Uncertainty Evaluation in Measurements and Machine Learning

---

Instructor: Miodrag Bolic, University of Ottawa, Ottawa, Canada<br/> www.site.uottawa.ca/~mbolic <br/>
Time and place:  Monday 8:30 - 11:30, MNO E218 <br/>
Course code: ELG 7172B (EACJ 5600)
<br>
### Course information

**Calendar Style Description:**
Uncertainty, Uncertainty propagation, Bayesian Inference, Bayesian Filtering, Data fusion, Metrology, Measurement Science, Error Analysis, Measures of Agreement, Data Quality, Data quality index.  Case studies will be drawn from various fields including biomedical instrumentation, sensors and signal processing.

**Prerequisites:** We expect participating students to bring basic knowledge and experience in
* Elementary Probability
* Elementary Statistics
* Signal processing
* Machine learning

**Grading:** For collecting the credits the student are expected to
* Assignments or projects (30% of the grade)
* Midterm (15% of the grade)
* Scribing (20% of the grade)
* Final exam (35% of the grade)

All lectures are given online using Zoom this year.

**About the course**
Over the last several years, deep neural networks advanced many applications including vision, language understanding, speech understanding, robotics and so on. But a major challenge still remains and that is: how to modeling uncertainty. Good models of uncertainty are crucial whenever decision needs to be made or an algorithm needs to decide how and when to acquire new information.
Uncertainty quantification is related to combining computational models, physical observations, and possibly expert judgment to make inferences about a physical system. Types of uncertainties include:
-	Experimental uncertainty (measurement errors)
-	Model uncertainty/discrepancy.
-	Input/parameter uncertainty.
-	Prediction uncertainty.

Why uncertainty:
- Uncertainty quantification is a fundamental component of model validation
- The objective is to replace the subjective notion of confidence with a mathematical rigorous measure
- Uncertainties relate to the physics of the problem of interest and not to the errors in the mathematical description/solution.







---
### Introduction

**Lecture **  
[Intro slides](/Lec01/Intro.pdf)



---
### Topic 1 Probabilistic reasoning
Reading: [Appendix D Probability by Kevin Murphy](https://probml.github.io/pml-book/book1.html) \
[A Comprehensive Tutorial to Learn Data Science with Julia from Scratch](https://www.analyticsvidhya.com/blog/2017/10/comprehensive-tutorial-learn-data-science-julia-from-scratch/)

**Lecture **  
[Prof. Rai's slides Lec1 9 t0 26](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec1_slides_print.pdf) \
[Prof. Rai's slides Lec2](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec2_slides_print.pdf)

**Exercise problems**  

---
### Topic 2 Bayesian Inference for Gaussians, Bayesian linear and logistic regression

Reading:
Code: [Continuous Data and the Gaussian Distribution](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Gaussian-Distribution.ipynb) \
[Bayesian linear regression](https://turing.ml/dev/tutorials/5-linearregression/) \
[Bayesian logistic regression](https://turing.ml/dev/tutorials/2-logisticregression/)

**Lecture **  
[Prof. Rai's slides Lec3](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec3_slides_print.pdf) \
[Prof. Rai's slides Lec4](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec4_slides_print.pdf) \
[Prof. Rai's slides Lec5](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec5_slides_print.pdf) \
[Prof. Rai's slides Lec6](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec6_slides_print.pdf)

**Exercise problems**  

---
### Topic 3.1 Monte Carlo, MCMC
Reading: \
Code: [MCMC PyTorch](deepbayes-2019/seminars/day5/)

**Lecture **  



**Exercise problems**  

---

### Topic 3.2 -  Variational Inference

Reading: [Blei et al JASA](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2017.1285773#.XraDPXUzaLI) | [Tran's VI Notes](/Material/VBnotesMNT.pdf) \
Other material: [Natural gradient notes](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) | [autograd in python](https://github.com/HIPS/autograd) | [ForwardDiff in Julia](https://github.com/JuliaDiff/ForwardDiff.jl) \
Code: [SVI in Pyro](https://pyro.ai/examples/svi_part_i.html)

**Lecture**  
[slides]()  


**Lab Topic 3**  


---

### Topic 3.3 Probabilistic programming

Reading:

**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 4 Bayesian models: hierarchical and mixture models
Reading: [Prior Choice Recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)

**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 5 - Gaussian Processes Regression and Classification

Reading:  [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) - Chapters 1, 2.1-2.5, 3.1-3.4, 3.7, 4.1-4.3. \
Code: [GP Stheno Python and Julia](https://github.com/wesselb/stheno) | [GPy for Python](https://sheffieldml.github.io/GPy/) | [Gausspr in R](https://rdrr.io/cran/kernlab/man/gausspr.html) | [Gaussianprocesses.jl in Julia](https://github.com/STOR-i/GaussianProcesses.jl) | [GPyTorch - GPs in PyTorch](https://gpytorch.ai/) \
Other material: [Visualize GP kernels](http://www.it.uu.se/edu/course/homepage/apml/GP/)


**Lecture**  
[slides]()  




---

### Topic 6 - Bayesian Model Inference

Reading (ordered by priority): [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf) - Chapter 7
| [Bayesian predictive methods article](https://link.springer.com/article/10.1007/s11222-016-9649-y) | [LOO-CV and WAIC article](https://link.springer.com/article/10.1007/s11222-016-9696-4) | [Bayesian regularization and Horseshoe](https://onlinelibrary-wiley-com.ezp.sub.su.se/doi/full/10.1002/wics.1463) | [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) - Chapters 5.1-5.4  

**Lecture**  



**Lab Topic 4**  

---

### Topic 7 Clustering with uncertainties: probabilistic PCA, VAR, normalizing flows
Reading: \
Code: [Vae with MNIST](https://github.com/bayesgroup/deepbayes-2019/tree/master/seminars/day2) | [Normalizing flow](https://github.com/bayesgroup/deepbayes-2019/tree/master/seminars/day3/nf)


**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 8 Time series models, forecasting and classification

Reading:\
Code: [ForneyLab](https://github.com/biaslab/ForneyLab.jl), [ForneyLab Documentation](https://biaslab.github.io/ForneyLab.jl/stable/)

**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 9 Sensor fusion

Reading:

**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 10 Integration of physical and machine learning models
Code: [SciMLTutorials.jl: Tutorials for Scientific Machine Learning and Differential Equations](https://github.com/SciML/SciMLTutorials.jl)
Reading:

**Lecture 1**  
[slides]()  

**Exercise problems**  

---

### Topic 11 Sequential decision making

Reading: [Algorithms for Decision Making](https://algorithmsbook.com/)

**Lecture 1**  
[slides]()  

**Exercise problems**  

---



### Links

* [This course at Github](https://github.com/Health-Devices/Course-Uncertanty-Machine-Learning-2021) <br>
* [Bright Space at the University of Ottawa](https://idp3.uottawa.ca/idp/login.jsp?actionUrl=%2Fidp%2FAuthn%2FUserPassword)
#### Relevant courses
* [Bayesian Learning](https://dev.deep-teaching.org/courses/bayesian-learning) by deep.TEACHING project
* [Bayesian Machine Learning and Information Processing (5SSD0)](https://biaslab.github.io/teaching/bmlip/) by Prof.dr.ir. Bert de Vries
* [Advanced Bayesian Learning](https://github.com/mattiasvillani/AdvBayesLearnCourse) by Mattias Villani
* [Probabilistic Machine Learning (Summer 2020)](https://uni-tuebingen.de/en/180804)

#### Books
* [Model-Based Machine Learning](https://www.mbmlbook.com/) by John Winn


### Support or Contact

[Miodrag Bolic email ](mailto:mbolic@site.uottawa.ca)
