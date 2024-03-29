# Uncertainty Evaluation in Engineering Measurements and Machine Learning

---

Instructor: Miodrag Bolic, University of Ottawa, Ottawa, Canada<br/> www.site.uottawa.ca/~mbolic <br/>
Time and place:  Winter 2022, Monday 8:30 - 11:30 <br/>
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
Over the last several years, deep neural networks advanced many applications including vision, language understanding, speech understanding, robotics and so on. But a major challenge still remains and that is: how to model uncertainty. Good models of uncertainty are crucial whenever decisions need to be made or an algorithm needs to decide how and when to acquire new information.
Uncertainty quantification is related to combining computational models, physical observations, and possibly expert judgment to make inferences about a physical system. Types of uncertainties include:
-	Experimental uncertainty (measurement errors)
-	Model uncertainty/discrepancy
-	Input/parameter uncertainty
-	Prediction uncertainty.

Why uncertainty:
- Uncertainty quantification is a fundamental component of model validation
- The objective is to replace the subjective notion of confidence with a mathematical rigorous measure
- Uncertainties relate to the physics of the problem of interest and not to the errors in the mathematical description/solution.


### Topics
PART 1: IID models and Bayesian inference
- Lec 1:  Introduction to modeling, MLE and MAP, Beta-binomial model 
- Lec 2: Linear Gaussian model, Bayesian linear regression, Logistic regression
- Lec 3a: Inference: Sampling: Rejection sampling, Importance sampling , Markov chain, MCMC: Metropolis Hastings, Gibbs sampling, Langevin dynamics, Stochastic gradient descent
- Lec 4: Variational inference, Black box variational inference
- Lec 5a: Uncertainty propagation and sensitivity analysis
- Lec 5b: Bayesian models: Mixture models, Hierarchical models
- Lec 6a: Model checking, model selection, Bayesian averaging, Information criteria
- Lec 6b: Expending traditional models: Introducing errors in both x and y variables, Bayesian neural networks

PART 2: Generative models, time series, heterogeneous data and decision making
- Lec 8: Gaussian Processes Regression and Classification
- Lec 9a: Sequential latent models: HMM, Kalman and particle filters
- Lec 9b: Deep sequential models:
- Lec 10: RNN, Sensor fusion
- Lec 11: Sequential decision making
- Lec 12: Scientific machine learning


---
[Binder](https://mybinder.org/v2/gh/Health-Devices/Course-Uncertanty-Machine-Learning-2021/HEAD)

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
### [Topic 3 Monte Carlo, MCMC](/Lec03)
**Lecture**  
Prof. Rai's slides [Lec15](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec15_slides_print.pdf), [Lec16](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec16_slides_print.pdf), [Lec17](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec17_slides_print.pdf) \
[Lec 3 Notebook](/Lec03/Lec3a.ipynb) \


**Additional Material**\
Reading:\
C. Andrieu, "[Monte Carlo Methods for Absolute Beginners](https://link.springer.com/content/pdf/10.1007/978-3-540-28650-9_6.pdf)," book chapter in Advanced Lectures on Machine Learning. pp. 113-145, 2003.\
Michael Betancourt, "[A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)," arXiv:1701.02434v2 [stat.ME] 16 Jul 2018. \
S. Callh, [Bayesian inference with Stochastic Gradient Langevin Dynamics](https://sebastiancallh.github.io/post/langevin/) \

Videos:\
[Hamiltonian Monte Carlo](https://www.youtube.com/watch?v=550ZHxodfg0&ab_channel=Applied_Bayesian_Stats), Applied Bayesian Statistics course, 2020.

**Exercise problems**  
Additional Material


---

### Topic 4 -  Variational Inference
Prof. Rai's slides [Lec13](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec13_slides_print.pdf),  [Lec14](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec14_slides_print.pdf) \
Variational Inference: Foundations and Modern Methods by David Blei, Rajesh Ranganath, Shakir Mohamed, [slides 55-81](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf) \
[Tutorials](https://rpubs.com/cakapourani) by  Chantriolnt-Andreas Kapourani

Code: [Turing Variational Inference](https://turing.ml/dev/tutorials/9-variationalinference/)

**Additional Material**
Reading: [Blei et al JASA](https://amstat.tandfonline.com/doi/abs/10.1080/01621459.2017.1285773#.XraDPXUzaLI) | [Tran's VI Notes](/Material/VBnotesMNT.pdf) \
Other material: [Natural gradient notes](https://wiseodd.github.io/techblog/2018/03/14/natural-gradient/) | [autograd in python](https://github.com/HIPS/autograd) | [ForwardDiff in Julia](https://github.com/JuliaDiff/ForwardDiff.jl) \
Code: [SVI in Pyro](https://pyro.ai/examples/svi_part_i.html)



---

### Topic 4.1 Probabilistic programming
Slides: [Probabilistic Machine Learning with PyMC3: Statistical Modeling for Engineers by Dr. Thomas Wiecki](/Lec04/Probabilistic_Programming.pptx)

Reading: [Probabilistic programming by Dan MacKinlay](https://danmackinlay.name/notebook/probabilistic_programming.html) \
Christopher Krapu, Mark Borsuk, "[Probabilistic programming: A review for environmental modellers](http://www.sciencedirect.com/science/article/pii/S1364815218308843)," Environmental Modelling & Software, Volume 114, 2019, Pages 40-48.
https://doi.org/10.1016/j.envsoft.2019.01.014.

---

### Topic 5a [Uncertainty propagation and sensitivity analysis](/Lec05)
Notebooks:
- [Lec 4a Uncertainty](/Lec04/Uncertainty.ipynb)
- [Lec 4a Sensitivity](/Lec04/Sensitivity.ipynb)

**Lecture**  
- [Lec 4a Uncertainty Slides](/Lec04/Uncertainty.ipynb)
- [Lec 4a Sensitivity Slides](/Lec04/Uncertainty.ipynb)

Julia Code:
- [Uncertainty Programming, Generalized Uncertainty Quantification](https://book.sciml.ai/lecture19/uncertainty_programming) by Chris Rackauckas
- [Global Sensitivity Analysis](https://book.sciml.ai/lecture17/global_sensitivity)  by Chris Rackauckas

Python code:
- [A practical introduction to sensitivity analysis](https://github.com/lrhgit/uqsa_tutorials/blob/master/sensitivity_introduction.ipynb) by Leif Rune Hellevik\
- [SALib - Sensitivity Analysis Library in Python](http://salib.readthedocs.io/en/latest/index.html)

Additional reading:
- V. G. Eck, W. P. Donders, J. Sturdy, J. Feinberg, T. Delhaas, L. R. Hellevik, and W. Huberts. A guide to uncertainty quantification and sensitivity analysis for cardiovascular applications. Int J Numer Method Biomed Eng, 32(8):e02755, 2016.


---

### Topic 5b [Bayesian models: hierarchical and mixture models](/Lec05)
[Lec 2 Notebook](/Lec02/Lec%202%20Linear%20models.ipynb) \

**Lecture**  
[Hierarchical models slides by Taylor R. Brown](https://github.com/tbrown122387/stat_6440_slides/blob/master/5/5.pdf) \
[Mixture Models slides by Russ	Salakhutdinov](http://www.cs.toronto.edu/~rsalakhu/sta4273_2013/notes/Lecture5_2013.pdf)

Code: [Turing Mixture Models](https://turing.ml/dev/tutorials/1-gaussianmixturemodel/) \
[PyMC3: A Primer on Bayesian Methods for Multilevel Modeling](https://docs.pymc.io/notebooks/multilevel_modeling.html#Example:-Radon-contamination-(Gelman-and-Hill-2006))\

Video:
[Introduction to Bayesian Multilevel models](https://www.youtube.com/watch?v=oGgJhOOnzZU&ab_channel=Applied_Bayesian_Stats)



---

### Topic 6.1 - Bayesian Model Checking and Selection
[Model checking by Taylor](https://github.com/tbrown122387/stat_6440_slides/blob/master/6/6.pdf)\
[Evaluating, comparing and expanding models by Taylor]https://github.com/tbrown122387/stat_6440_slides/blob/master/7/7.pdf

Reading (ordered by priority): [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/BDA3.pdf) - Chapters 6 and 7
| [Bayesian Model Selection, Model Comparison, and Model Averaging article](https://fhollenbach.org/papers/Hollenbach_Montgomery_2019_BayesianModelSelection.pdf)

**Lecture**  

### Topic 6.2 - Expanding traditional models
#### [Expanding neural networks - Bayesian neural networks](/Lec06)
Slides: [Probabilistic Modeling meets Deep Learning by Prof. Rai slides 1-9](https://www.cse.iitk.ac.in/users/piyush/courses/tpmi_winter19/tpmi_w19_lec23_slides_print.pdf)\
[Stochastic variational inference and Bayesian neural networks by Nadezhda Chirkova](https://indico.cern.ch/event/845380/sessions/323554/attachments/1952856/3250850/bnn_tutorial.pdf)

Reading: [An introduction to Bayesian neural networks](https://engineering.papercup.com/posts/bayesian-neural-nets/) \
[Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users](https://arxiv.org/abs/2007.06823) and [video](https://www.youtube.com/watch?v=T5TPaI5H4q8&t=579s&ab_channel=MohammedBennamoun)

Code: [Turing Julia Bayesian Neural Networks](https://turing.ml/dev/tutorials/3-bayesnn/)

---

---

### Topic 8 - Gaussian Processes Regression and Classification
**Lecture**  
Slides by Andreas Lindholm [1](https://uppsala.instructure.com/courses/28106/pages/lecture-7-gaussian-processes-i?module_item_id=64155) and [2](https://uppsala.instructure.com/courses/28106/pages/lecture-8-gaussian-processes-ii?module_item_id=64157)  

Reading:  [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf) - Chapters 1, 2.1-2.5, 3.1-3.4, 3.7, 4.1-4.3. \
Code: [GP Stheno Python and Julia](https://github.com/wesselb/stheno) | [GPy for Python](https://sheffieldml.github.io/GPy/) | [GP summer school - Labs in Python](http://gpss.cc/gpss20/labs) | [GP in Turing Regression](https://luiarthur.github.io/TuringBnpBenchmarks/gp) | [GP in Turing Classification](https://luiarthur.github.io/TuringBnpBenchmarks/gpclassify)
Other material: [Visualize GP kernels](http://smlbook.org/GP/) \
[More slides](/Lec06/epsrcws08_rasmussen_lgp_01.pdf)  

---

### Topic 9 Sequential latent models
Slides: [HMM by John Paisley](/Lec09/HMM_Paisley.pdf) \
[ELEC-E8105 - Non-linear Filtering and Parameter Estimation by Simo Sarkaa](https://mycourses.aalto.fi/course/view.php?id=20984&section=2), Lec 1, 2, 3, and 6 \
[Particle filters](/Lec09/PFs.ppt)

Reading: [Tutorial paper on latent-variable models for time-series data](http://web4.cs.ucl.ac.uk/staff/d.barber/publications/GMSPM.pdf), Barber and Cemgil, IEEE Signal Processing Magazine, 2010. \
[Kinematic Models for Target Tracking](https://webee.technion.ac.il/people/shimkin/Estimation09/ch8_target.pdf) | [Markov Models From The Bottom Up, with Python](https://ericmjl.github.io/essays-on-data-science/machine-learning/markov-models/)\
[Elements of Sequential Monte Carlo](https://arxiv.org/pdf/1903.04797.pdf)

Code: [ForneyLab](https://github.com/biaslab/ForneyLab.jl), [ForneyLab Documentation](https://biaslab.github.io/ForneyLab.jl/stable/) \
[LowLevelParticleFilters](https://github.com/baggepinnen/LowLevelParticleFilters.jl) \
[Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)\

---


### Topic 10a Sensor fusion
Slides:  [Multisensor Data Fusion : Techno Briefing](https://www.slideshare.net/paveenju/2014-data-fusionpptx) \
[Introduction to Data Association by B. Collins](http://www.cse.psu.edu/~rtc12/CSE598C/datassocPart1.pdf)  
[Lec 1 of the course MultiModal Machine Learning by Prof Morency](https://cmu-multicomp-lab.github.io/mmml-course/fall2020/schedule/) \
[SF Course by Prof. Gustafsson](http://sensorfusion.se/sf-course/) \
[Penn State Stats 505 CCA example](https://online.stat.psu.edu/stat505/lesson/13)

Reading: [A Review of Data Fusion Techniques by F. Castanedo](https://www.hindawi.com/journals/tswj/2013/704504/) \
[Multimodal Machine Learning: A Survey and Taxonomy by T. Baltrusaitis](https://arxiv.org/abs/1705.09406)

### Topic10b Deep sequential models
Slides:
[DEEP SEQUENTIAL LATENT VARIABLE MODELS by J. Marino](https://joelouismarino.github.io/files/lectures/2019/deep_sequential_models.pdf) slides 31-49 \
[Lecture 6: Recurrent Neural Networks by Ming Li](https://cs.uwaterloo.ca/~mli/Deep-Learning-2017-Lecture6RNN.ppt) \
[Generative modelling with deep latent variable models  by M. Fraccaro](http://summer-school-gan.compute.dtu.dk/slides_marco.pdf) slides 51-91.

Resources: [Deep Learning book, Chapter 10: Sequence Modeling: Recurrent and Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html), [Video lecture](https://www.youtube.com/watch?v=ZVN14xYm7JA&ab_channel=AlenaKruchkova) \
[Deep Learning Time Series Forecasting](https://github.com/Alro10/deep-learning-time-series)\
[Deep Latent Variable Models for Sequential Data by M. Fraccaro](https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf)

---

### [Topic 11 Sequential decision making](/Lec11)
Slides: [Reinforcement Learning and Multi-arm Bandits by Dr. Ravindran](/Lec11/MDP_Class_22.pdf) |
[Lecture and slides from deepmind](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)

Videos: [Reinforcement learning course](https://www.cse.iitm.ac.in/~ravi/courses/Reinforcement%20Learning.html)

Reading: [Algorithms for Decision Making](https://algorithmsbook.com/) Chapters 15, 16, 17 | [Paper on playing atari games](https://arxiv.org/pdf/1312.5602v1.pdf)

Code: [Julia Academy Decision Making](https://htmlview.glitch.me/?https://github.com/JuliaAcademy/Decision-Making-Under-Uncertainty/blob/master/html/1-MDPs.jl.html) |
[REINFORCEjs](https://cs.stanford.edu/people/karpathy/reinforcejs/waterworld.html)


---

### [Topic 12a Integration of physical and machine learning models](/Lec12)
Slides: [Integrating Scientific Theory with Machine Learning](/Lec11/ScientificMachineLearning.pdf)

Code: [SciMLTutorials.jl: Tutorials for Scientific Machine Learning and Differential Equations](https://github.com/SciML/SciMLTutorials.jl)\
[Introduction to Scientific Machine Learning through Physics-Informed Neural Networks, Chris Rackauckas](https://book.sciml.ai/notes/03/)\
[Bayesian Estimation of Differential Equations](https://turing.ml/dev/tutorials/10-bayesian-differential-equations/)\
[ADCME](https://juliahub.com/docs/ADCME/b8Ld2/0.5.7/)

Video: [COVID-19 Epidemic Mitigation via Scientific Machine Learning SciML](https://www.youtube.com/watch?v=jMhPZFZ0yvE&t=3108s&ab_channel=ChristopherRackauckas) \
[Code: Modeling COVID 19 with Differential Equations](https://julia.quantecon.org/continuous_time/seir_model.html)

---


---

### [Topic 12b Conclusion](/Lec12)
Slides: [Review](/Lec12/Review.pptx)

Reading: [Learning with non-IID data +other ML assumptions and how to break them - Tegan Maharaj](https://sites.google.com/mila.quebec/ift6135/lectures)

---



### Links

* [This course at Github](https://github.com/Health-Devices/Course-Uncertanty-Machine-Learning-2021) <br>
* [Bright Space at the University of Ottawa](https://idp3.uottawa.ca/idp/login.jsp?actionUrl=%2Fidp%2FAuthn%2FUserPassword)
#### Relevant courses
* [Bayesian Learning](https://dev.deep-teaching.org/courses/bayesian-learning) by deep.TEACHING project
* [Bayesian Machine Learning and Information Processing (5SSD0)](https://biaslab.github.io/teaching/bmlip/) by Prof.dr.ir. Bert de Vries
* [Advanced Bayesian Learning](https://github.com/mattiasvillani/AdvBayesLearnCourse) by Mattias Villani
* [Probabilistic Machine Learning (Summer 2020)](https://uni-tuebingen.de/en/180804)
* [Neuromatch Academy: Computational Neuroscience](https://compneuro.neuromatch.io/projects/docs/datasets_overview.html)

#### Books
* Murphy, Kevin P. 2021. Probabilistic Machine Learning: An Introduction, MIT press.
* Murphy, Kevin P. 2022, Probabilistic Machine Learning: Advanced Topics.
* Bayesian Reasoning and Machine Learning by David Barber.
* Bayesian Methods for Hackers by Cameron Davidson-Pilon.
* N. D. Goodman, J. B. Tenenbaum, and The ProbMods Contributors (2016). Probabilistic Models of Cognition (2nd ed.)
* [Model-Based Machine Learning](https://www.mbmlbook.com/) by John Winn
* Variational Methods for Machine Learning with Applications to Deep Networks by L.P. Cinelli et al., 2021.
* [Physics-based Deep Learning](http://physicsbaseddeeplearning.org) by N. Thuerey et al., Dec 2021.
* Statistics with Julia: Fundamentals for Data Science, Machine Learning and Artificial Intelligence, by Y. Nazarathy et al., 2021.
* Bayesian Modeling and Computation in Python by Martin, Osvaldo et al.,2021.
* An Introduction to Neural Network Methods for Differential Equations by Neha Yadav, Anupam Yadav, 2015.
* [Probabilistic Machine Learning for Civil Engineers](http://profs.polymtl.ca/jagoulet/Site/Goulet_web_page_BOOK.html) by James-A. Goulet, 2020.

### Support or Contact
[Miodrag Bolic email ](mailto:mbolic@site.uottawa.ca)
