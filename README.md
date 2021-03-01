# Uncertainty Evaluation in Engineering Measurements and Machine Learning

---

Instructor: Miodrag Bolic, University of Ottawa, Ottawa, Canada<br/> www.site.uottawa.ca/~mbolic <br/>
Time and place:  Monday 8:30 - 11:30, MNO E218 <br/>
Course code: ELG 5218 (EACJ 5600)
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
-	Model uncertainty/discrepancy
-	Input/parameter uncertainty
-	Prediction uncertainty.

Why uncertainty:
- Uncertainty quantification is a fundamental component of model validation
- The objective is to replace the subjective notion of confidence with a mathematical rigorous measure
- Uncertainties relate to the physics of the problem of interest and not to the errors in the mathematical description/solution.







---
### Introduction

**Lecture**  
Reading: Z. Ghahramani, “[Probabilistic machine learning and artificial intelligence](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/03/Ghahramani.pdf),” Nature, 2015.

[Intro slides](/Lec01/Intro.pdf)



---
### [Topic 1 Probabilistic reasoning](/Lec01)
Reading: [Appendix D Probability by Kevin Murphy](https://probml.github.io/pml-book/book1.html) \
[A Comprehensive Tutorial to Learn Data Science with Julia from Scratch](https://www.analyticsvidhya.com/blog/2017/10/comprehensive-tutorial-learn-data-science-julia-from-scratch/)

**Lecture**  
[Prof. Rai's slides Lec1 9 t0 26](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec1_slides_print.pdf) \
[Lec 1 Notebook](/Lec01/Bayesian%20Modeling%20in%20Julia.ipynb)

**Additional Material**\
[Prof. Rai's slides Lec2](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec2_slides_print.pdf)



---
### [Topic 2 Bayesian Inference for Gaussians, Bayesian linear and logistic regression](/Lec02)
**Lecture**  
[Lec 2 Notebook](/Lec02/Lec%202%20Linear%20models.ipynb) \
[Prof. Rai's slides Lec5](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec5_slides_print.pdf) \
[Prof. Rai's slides Lec6](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec6_slides_print.pdf)

**Mandatory Exercise problems**  
Rehearsal Exercises->Probability Theory Review, Bayesian Machine Learning, Continuous Data and the Gaussian Distribution, Regression from [Exercise: Bayesian Machine Learning and Information Processing](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises.ipynb) and [Solutions](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/exercises/Exercises-with-Solutions.ipynb)

**Additional Material**\
Reading:\
[Chapters 3.6 and 11 by Kevin Murphy](https://probml.github.io/pml-book/book1.html) \
[Prof. Rai's slides Lec3](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec3_slides_print.pdf) \
[Prof. Rai's slides Lec4](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec4_slides_print.pdf) \
Code: \
[Continuous Data and the Gaussian Distribution](https://nbviewer.jupyter.org/github/bertdv/BMLIP/blob/master/lessons/notebooks/The-Gaussian-Distribution.ipynb) \
[Bayesian linear regression](https://turing.ml/dev/tutorials/5-linearregression/) \
[Bayesian logistic regression](https://turing.ml/dev/tutorials/2-logisticregression/)

---
### [Topic 3.1 Monte Carlo, MCMC](/Lec02)
**Lecture**  
Prof. Rai's slides [Lec15](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec15_slides_print.pdf), [Lec16](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec16_slides_print.pdf), [Lec17](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec17_slides_print.pdf) \
[Lec 3 Notebook](/Lec03/Lec3a.ipynb) \
[Bayesian inference with Stochastic Gradient Langevin Dynamics](https://sebastiancallh.github.io/post/langevin/)

**Exercise problems**  
Additional Material


---

### Topic 3.2 -  Variational Inference
Prof. Rai's slides [Lec13](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec13_slides_print.pdf),  [Lec14](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec14_slides_print.pdf) \
Variational Inference: Foundations and Modern Methods by David Blei, Rajesh Ranganath, Shakir Mohamed, [slides 55-81](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf) \
[Tutorials](https://rpubs.com/cakapourani) by  Chantriolnt-Andreas Kapourani

Code: [Turing Variational Inference](https://turing.ml/dev/tutorials/9-variationalinference/)

**Additional Material**
Reading: [Blei et al JASA](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2017.1285773#.XraDPXUzaLI) | [Tran's VI Notes](/Material/VBnotesMNT.pdf) \
Other material: [Natural gradient notes](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) | [autograd in python](https://github.com/HIPS/autograd) | [ForwardDiff in Julia](https://github.com/JuliaDiff/ForwardDiff.jl) \
Code: [SVI in Pyro](https://pyro.ai/examples/svi_part_i.html)

**[Assignment 1](/Assignments/Assignment1.pdf)**  



---

### Topic 3.3 Probabilistic programming
Slides: [Probabilistic Machine Learning with PyMC3: Statistical Modeling for Engineers by Dr. Thomas Wiecki](/Lec04/Probabilistic_Programming.pptx)

Reading: [Probabilistic programming by Dan MacKinlay](https://danmackinlay.name/notebook/probabilistic_programming.html) \
Christopher Krapu, Mark Borsuk, "[Probabilistic programming: A review for environmental modellers](http://www.sciencedirect.com/science/article/pii/S1364815218308843)," Environmental Modelling & Software, Volume 114, 2019, Pages 40-48.
https://doi.org/10.1016/j.envsoft.2019.01.014.



---

### Topic 4 Bayesian models: hierarchical and mixture models
Reading: [Prior Choice Recommendations](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)

**Lecture**  
[Hierarchical models slides by Taylor R. Brown](https://github.com/tbrown122387/stat_6440_slides/blob/master/5/5.pdf) \
[Mixture Models slides by Russ	Salakhutdinov](http://www.cs.toronto.edu/~rsalakhu/sta4273_2013/notes/Lecture5_2013.pdf)

Code: [Turing Mixture Models](https://turing.ml/dev/tutorials/1-gaussianmixturemodel/) \
[PyMC3: A Primer on Bayesian Methods for Multilevel Modeling](https://docs.pymc.io/notebooks/multilevel_modeling.html#Example:-Radon-contamination-(Gelman-and-Hill-2006))


---

### Topic 5.1 - Bayesian Model Checking and Selection
[Model checking by Taylor](https://github.com/tbrown122387/stat_6440_slides/blob/master/6/6.pdf)\
[Evaluating, comparing and expanding models by Taylor]https://github.com/tbrown122387/stat_6440_slides/blob/master/7/7.pdf

Reading (ordered by priority): [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf) - Chapters 6 and 7
| [Bayesian Model Selection, Model Comparison, and Model Averaging article](https://fhollenbach.org/papers/Hollenbach_Montgomery_2019_BayesianModelSelection.pdf)

**Lecture**  

### Topic 5.2 - Expanding traditional models
#### Expending regression models with measurement errors
Slides: [Errors and uncertainty in variables by Stefanie Muff](http://www.biometrische-gesellschaft.de/fileadmin/AG_Daten/BayesMethodik/workshops_etc/2016-12_Mainz/Muff2016-slides.pdf) \
Reading: [Bayesian analysis of measurement errors](https://arxiv.org/pdf/1302.3065.pdf)\
[Example of measurement error effects](https://stefaniemuff.shinyapps.io/MEC_ChooseL/)


#### Expanding neural networks - Bayesian neural networks
Slides: [Probabilistic Modeling meets Deep Learning by Prof. Rai slides 1-9](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec23_slides_print.pdf)\
[Stochastic variational inference and Bayesian neural networks by Nadezhda Chirkova](https://indico.cern.ch/event/845380/sessions/323554/attachments/1952856/3250850/bnn_tutorial.pdf)

Reading: [An introduction to Bayesian neural networks](https://engineering.papercup.com/posts/bayesian-neural-nets/) \
[Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users](https://arxiv.org/abs/2007.06823) and [video](https://www.youtube.com/watch?v=T5TPaI5H4q8&t=579s&ab_channel=MohammedBennamoun)

Code: [Turing Julia Bayesian Neural Networks](https://turing.ml/dev/tutorials/3-bayesnn/)

**[Assignment 2](/Assignments/Assignment2.pdf), [Dataset](/Assignments/TrafficSpeed.csv)**  
**[Midterm Review](/Lec06/MidtermReview.pdf)**

### Topic 6 - Gaussian Processes Regression and Classification
**Lecture**  
Slides by Andreas Lindholm [1](https://uppsala.instructure.com/courses/28106/pages/lecture-7-gaussian-processes-i?module_item_id=64155) and [2](https://uppsala.instructure.com/courses/28106/pages/lecture-8-gaussian-processes-ii?module_item_id=64157)  

Reading:  [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) - Chapters 1, 2.1-2.5, 3.1-3.4, 3.7, 4.1-4.3. \
Code: [GP Stheno Python and Julia](https://github.com/wesselb/stheno) | [GPy for Python](https://sheffieldml.github.io/GPy/) | [GP summer school - Labs in Python](http://gpss.cc/gpss20/labs) | [GP in Turing Regression](https://luiarthur.github.io/TuringBnpBenchmarks/gp) | [GP in Turing Classification](https://luiarthur.github.io/TuringBnpBenchmarks/gpclassify)
Other material: [Visualize GP kernels](http://smlbook.org/GP/) \
[More slides](/Lec06/epsrcws08_rasmussen_lgp_01.pdf)  

### Topic 7 Generative models" probabilistic PCA, Variational autoencoders
Linear latent variable models: \
Reading: [Chapter 2 from A.M. O’Sullivan's thesis](https://spiral.imperial.ac.uk/bitstream/10044/1/19191/3/OSullivan-AM-2013-PhD-Thesis.pdf)\
Slides:  [Continuous Latent Variable Models by Russ Salakhutdinov](http://www.cs.utoronto.ca/~rsalakhu/STAD68/notes/Lecture9_2014.pdf)
Code: [MultivariateStats](https://multivariatestatsjl.readthedocs.io/en/stable/index.html#), [Turing example of pPCA](https://github.com/TuringLang/TuringExamples/blob/master/probabilisticpca.jl)

Variational autoencoders: \
Reading: [An Introduction to Variational Autoencoders by D. P. Kingma](https://arxiv.org/abs/1906.02691) please read chapters 1 and 2 , | [From Autoencoder to Beta-VAE by Lilian Weng](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html#td-vae) \
Slides: [Variational autoencoders by Prof.Mitesh Khapra](https://www.cse.iitm.ac.in/~miteshk/CS7015/Slides/Teaching/pdf/Lecture21.pdf) \
Videos: [Variational autoencoders by Prof.Mitesh Khapra](https://freevideolectures.com/course/4194/nptel-deep-learning-part/30) \
Code: [Autoencoders by  Hugh Murrell](https://nextjournal.com/DeepLearningNotes/Ch11Autoencoders?change-id=CWrxF3xs45ozzPvxyPcapr&node-id=63634d84-e1ba-4ec0-8ea0-53549e4be0b3)

---

### Topic 8 Time series models, forecasting and classification

Reading:\
Code: [ForneyLab](https://github.com/biaslab/ForneyLab.jl), [ForneyLab Documentation](https://biaslab.github.io/ForneyLab.jl/stable/)

**Lecture**  
[slides]()  

**Exercise problems**  

---

### Topic 9 Sensor fusion

Reading:


**Code**
[BayesianDataFusion.jl](http://jaak-s.github.io/BayesianDataFusion.jl/#bayesiandatafusionjl-package)

---

### Topic 10 Integration of physical and machine learning models
Code: [SciMLTutorials.jl: Tutorials for Scientific Machine Learning and Differential Equations](https://github.com/SciML/SciMLTutorials.jl)\
[ADCME](https://juliahub.com/docs/ADCME/b8Ld2/0.5.7/)
Reading:



**Exercise problems**  

---

### Topic 11 Sequential decision making

Reading: [Algorithms for Decision Making](https://algorithmsbook.com/)

**Lecture**  
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
