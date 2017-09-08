# Single view hand pose estimation

## Table of Contents
 - [arXiv Papers](#arxiv-papers)
 - [Conference Papers](#conference-papers)
   - [2017 CVPR](#2017-cvpr)
   - [2017 Others](#2017-others)
   - [2016 ECCV](#2016-eccv)
   - [2016 CVPR](#2016-cvpr)
   - [2016 Others](#2016-others)
   - [2015 ICCV](#2015-iccv)
   - [2015 CVPR](#2015-cvpr)
   - [2015 Others](#2015-others)
   - [2014 CVPR](#2014-cvpr)
   - [2014 Others & Before](#2014-others--before)
 - [Journal Papers](#journal-papers)
 - [Theses](#theses)
 - [Other Related Papers](#other-related-papers)

## Evaluation codes
See folder [``evaluation``] to get more details about performance evaluation for hand pose estimation.

## arXiv Papers
##### [\[arXiv:1708.03416\]](https://arxiv.org/abs/1708.03416) Pose Guided Structured Region Ensemble Network for Cascaded Hand Pose Estimation. [\[PDF\]](https://arxiv.org/pdf/1708.03416.pdf)
_Xinghao Chen, Guijin Wang, Hengkai Guo, Cairong Zhang_
![Alt text](media/arxiv1708-structured-region-ensemble.png?raw=true "Optional Title")
-   using an iterative refinement procedure, which extract  feature  regions  under  the  guide  of  hand  pose from previous stage.
-   we propose a hierarchical method to fuse features of different joints  according to the topology of hand.

##### [\[arXiv:1707.07248\]](https://arxiv.org/abs/1707.07248) Towards Good Practices for Deep 3D Hand Pose Estimation. [\[PDF\]](https://arxiv.org/pdf/1707.07248.pdf) [\[Code\]](https://github.com/guohengkai/region-ensemble-network)
_Hengkai Guo, Guijin Wang, Xinghao Chen, Cairong Zhang_
![Alt text](media/arxiv1707-pose-estimation-practices.png?raw=true "Optional Title")
-   single deep ConvNet architecture named Region Ensemble Net (REN) to directly regress the 3D hand joint coordinates with end-to-end optimization and inference.
-   residual connection, data augmentation and smooth L1 loss.

##### [\[arXiv:1707.02237\]](https://arxiv.org/abs/1707.02237) The 2017 Hands in the Million Challenge on 3D Hand Pose Estimation. [\[PDF\]](https://arxiv.org/pdf/1707.02237.pdf)
_Shanxin Yuan, Qi Ye, Guillermo Garcia-Hernando, Tae-Kyun Kim_
![Alt text](media/arxiv1707-large-multiview-dataset.png?raw=true "Optional Title")

##### [\[arXiv:1705.09606\]](https://arxiv.org/abs/1705.09606) End-to-end Global to Local CNN Learning for Hand Pose Recovery in Depth data. [\[PDF\]](https://arxiv.org/pdf/1705.09606.pdf)
_Meysam Madadi, Sergio Escalera, Xavier Baro, Jordi Gonzalez_
![Alt text](media/arxiv1705-end2end-global2local.png?raw=true "Optional Title")
 - So we break the hand pose estimation problem into hierarchical optimization subtasks, each one focused on a specific finger and hand region.
 - In addition, we model correlated motion among fingers through fully connected layers and training the whole network in an end-to-end fashion.

##### [\[arXiv:1705.01389\]](https://arxiv.org/abs/1705.01389) Learning to Estimate 3D Hand Pose from Single RGB Images. [\[PDF\]](https://arxiv.org/pdf/1705.01389.pdf)  [\[Project Page\]](https://lmb.informatik.uni-freiburg.de/projects/hand3d/)   [\[Code\]](https://github.com/lmb-freiburg/hand3d)
_Christian Zimmermann, Thomas Brox_
![Alt text](media/arxiv1705-learn-3d-hand-pose.png?raw=true "Optional Title")
The third network finally derives the 3D hand pose from the 2D keypoints.

##### [\[arXiv:1704.02224\]](https://arxiv.org/abs/1704.02224) Hand3D: Hand Pose Estimation using 3D Neural Network. [\[PDF\]](https://arxiv.org/pdf/1704.02224.pdf)  [\[Project Page\]](http://www.idengxm.com/hand3d/index.html)
_Xiaoming Deng\*, Shuo Yang\*, Yinda Zhang\*, Ping Tan, Liang Chang, Hongan Wang_
![Alt text](media/arxiv1704-3dnn-pose.png?raw=true "Optional Title")
 - Our  method  does  not  rely  on  any  predefined model and require no post-processing for 2D/3D projection which may potentially increase error.
 - We perform data augmentation to increase both the quality and quantity of
the training data.

##### [\[arXiv:1704.02201\]](https://arxiv.org/abs/1704.02201) Real-time Hand Tracking under Occlusion from an Egocentric RGB-D Sensor. [\[PDF\]](https://arxiv.org/pdf/1704.02201.pdf)
_Franziska Mueller, Dushyant Mehta, Oleksandr Sotnychenko, Srinath Sridhar, Dan Casas, Christian Theobalt_
![Alt text](media/arxiv1704-occlusion-egocentric.png?raw=true "Optional Title")
 - localizes the hand and estimates, in  real  time,  the  3D  joint  locations  from  egocentric viewpoints, in clutter, and under strong occlusions using two CNNs.
 - A photorealistic data generation framework for synthesizing large amounts of annotated RGB-D training data of hands in natural interaction with objects and clutter.
 - Extensive  evaluation  on  a  new  annotated  egocentric benchmark dataset featuring cluttered scenes and interaction with objects.

##### [\[arXiv:1612.00596\]](https://arxiv.org/abs/1612.00596) Learning to Search on Manifolds for 3D Pose Estimation of Articulated Objects. [\[PDF\]](https://arxiv.org/pdf/1612.00596.pdf)
_Yu Zhang, Chi Xu, Li Cheng_
![Alt text](media/arxiv1612-articulated-pose-manifold.png?raw=true "Optional Title")
-   an extension of L2S that typically operates in discrete output spaces to continuous output spaces.
-   Lie-group output space (points on manifolds expressed as rigid-transformation Lie-groups) is considered by working with 3D skeletal model.


## Conference Papers

### 2017 CVPR
##### Hand Keypoint Detection in Single Images using Multiview Bootstrapping. [\[PDF\]](https://arxiv.org/pdf/1704.07809) [\[Project Page\]](http://www.cs.cmu.edu/~tsimon/projects/mvbs.html) [\[Code\]](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
_Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh_
![Alt text](media/cvpr17-multiview-bootstrapping.png?raw=true "Optional Title")
-    In particular, it allows a weak detector, trained on  a  small  annotated  dataset,  to  localize  subsets  of  keypoints  in _good_ views  and  uses  robust  3D  triangulation  to filter out incorrect detections.
-   produces hand keypoint detectors for RGB images that rival the performance of RGB-D hand keypoint detectors.
-   applying this single view detector in a multicamera setup allows markerless 3D hand reconstructions in unprecedented scenario.

##### Crossing Nets: Dual Generative Models with a Shared Latent Space for Hand Pose Estimation. [\[PDF\]](https://arxiv.org/pdf/1702.03431.pdf)
_Chengde Wan, Thomas Probst, Luc Van Gool, Angela Yao_
![Alt text](media/cvpr17-crossingnets-dual-generative-shared-latent.png?raw=true "Optional Title")
-   extend  the  GAN  to  a  semi-supervised  setting for real-valued structured prediction.
-   synthesize highly realistic and accurate depth maps of the articulated hand during training.
-   novel  distance  constraint  enforces smoothness  in  the  learned  latent  space  so  that  per- forming a random walk in the latent space corresponds
to synthesizing a sequence of realistically interpolated poses and depth maps.

##### Big Hand 2.2M Benchmark: Hand Pose Data Set and State of the Art Analysis. [\[PDF\]](https://arxiv.org/pdf/1704.02612.pdf)
_Shanxin Yuan, Qi Ye, Bjorn Stenger, Siddhand Jain, Tae-Kyun Kim_
![Alt text](media/cvpr17-big-hand-2.2m-benchmark.png?raw=true "Optional Title")

##### 3D Convolutional Neural Networks for Efficient and Robust Hand Pose Estimation from Single Depth Images.[\[PDF\]](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2017/3D%20Convolutional%20Neural%20Networks%20for%20Efficient%20and%20Robust%20Hand%20Pose%20Estimation%20from%20Single%20Depth%20Images.pdf) [\[Project Page\]](https://sites.google.com/site/geliuhaontu/home/cvpr2017)
_Liuhao Ge, Hui Liang, Junsong Yuan and Daniel Thalmann_
![Alt text](media/cvpr17-3dcnn-pose.png?raw=true "Optional Title")
-   3D  volumetric  representation  for hand pose estimation, which directly regress 3D joint locations from 3D features in a single pass without adopting any iterative refinement process.
-   over 215 fps on a single GPU, due to relatively shallow architecture for the 3D CNN.
-   3D data augmentation on the training set.


### 2016 ECCV
##### Spatial Attention Deep Net with Partial PSO for Hierarchical Hybrid Hand Pose Estimation. [\[PDF\]](http://www.iis.ee.ic.ac.uk/ComputerVision/docs/pubs/Qi_Shanxin_ECCV_2016.pdf) [\[Project Page\]](https://sites.google.com/site/qiyeincv/home/eccv2016)
_Qi Ye\*, Shanxin Yuan\*, Tae-Kyun Kim_
![Alt text](media/eccv16-spacial-attention-partial-pso-hierarchical.png?raw=true "Optional Title")
In this paper, a hybrid hand pose estimation method is proposed by applying the kinematic hierarchy strategy to the input space (as well as the output space) of the discriminative method by a spatial attention mechanism and to the optimization
of the generative method by hierarchical Particle Swarm Optimization (PSO).

##### Hand Pose Estimation from Local Surface Normals. [\[PDF\]](http://www.vision.ee.ethz.ch/~yaoa/pdfs/wan_eccv16.pdf)
_Chengde Wan, Angela Yao, and Luc Van Gool_
![Alt text](media/eccv16-local-surface-normal.png?raw=true "Optional Title")
-   the first to incorporate local surface normals for pose estimation.
-   extend the commonly used depth difference feature to an angular difference feature between two normal directions, which is highly robust to 3D rigid transformation.
-   propose a flexible conditional regression framework, encoding all previously estimated information as a part of the local reference frame.

##### Real-time Joint Tracking of a Hand Manipulating an Object from RGB-D Input. [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/content/RealtimeHO_ECCV2016.pdf) [\[Project Page\]](http://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/)
_Srinath Sridhar, Franziska Mueller, Michael Zollhöfer, Dan Casas, Antti Oulasvirta, Christian Theobalt_
![Alt text](media/eccv16-realtime-hand-manipulating.png?raw=true "Optional Title")
-   3D articulated Gaussian mixture alignment approach.
-   Novel contact point and occlusion objective terms that were motivated by
the physics of grasps.
-   multi-layered classification architecture to segment hand and object, and classify hand parts.

### 2016 Others
##### \[2016 ICPR\] Depth-based 3D hand pose tracking. [\[PDF\]](http://ieeexplore.ieee.org/abstract/document/7900051)
_Kha Gia Quach, Chi Nhan Duong, Khoa Luu, and Tien D. Bui._

##### \[2016 IJCAI\] Model-based Deep Hand Pose Estimation. [\[PDF\]](http://xingyizhou.xyz/zhou2016model.pdf) [\[Project Page\]](http://xingyizhou.xyz/) [\[Code\]](https://github.com/tenstep/DeepModel)
_Xingyi Zhou, Qingfu Wan, Wei Zhang, Xiangyang Xue, Yichen Wei_

##### \[2016 SIGGRAPH\] Efficient and precise interactive hand tracking through joint, continuous optimization of pose and correspondences. [\[PDF\]](http://www.cs.toronto.edu/~jtaylor/papers/SIGGRAPH2016-SmoothHandTracking.pdf)
_Jonathan Taylor et al._

##### \[2016 SIGGRAPH Asia\] Sphere-Meshes for Real-Time Hand Modeling and Tracking. [\[PDF\]](http://lgg.epfl.ch/publications/2016/HModel/paper.pdf)  [\[Project Page\]](http://lgg.epfl.ch/publications/2016/HModel/index.php) [\[Code\]](https://github.com/OpenGP/hmodel)
_Anastasia Tkach, Mark Pauly, Andrea Tagliasacchi_

### 2016 CVPR
##### Robust 3D Hand Pose Estimation in Single Depth Images: From Single-View CNN to Multi-View CNNs. [\[PDF\]](http://eeeweba.ntu.edu.sg/computervision/Research%20Papers/2016/Robust%203D%20Hand%20Pose%20Estimation%20in%20Single%20Depth%20Images,%20from%20Single-View%20CNN%20to%20Multi-View%20CNNs.PDF) [\[Project Page\]](https://sites.google.com/site/geliuhaontu/home/cvpr2016) [\[Code\]](https://github.com/geliuhao/CVPR2016_HandPoseEstimation)
_Liuhao Ge, Hui Liang, Junsong Yuan, Daniel Thalmann_
![Alt text](media/cvpr16-multiview-cnn.png?raw=true "Optional Title")
-   generate heat-maps for front, side and top views simultaneously, from which the 3D locations of hand joints can be estimated more robustly.
-   heat-maps from other two views can help to eliminate the ambiguity.
-   embeds hand pose constraints learned from training samples in an implicit way, which allows to enforce hand motion constraints without manually defining hand size parameters.

##### DeepHand: Robust Hand Pose Estimation by Completing a Matrix Imputed With Deep Features.  [\[PDF\]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Sinha_DeepHand_Robust_Hand_CVPR_2016_paper.pdf)[\[Project Page\]](https://engineering.purdue.edu/cdesign/wp/deephand-robust-hand-pose-estimation/)
_Ayan Sinha\*, Chiho Choi\*, Karthik Ramani_
![Alt text](media/cvpr16-deephand-completing-matrix.png?raw=true "Optional Title")
-   Initialization of the pose matrix using a low dimensional and discriminative representation, which aids efficient retrieval of nearest neighbors from a large population of pre-computed activation features.
-   An efficient matrix completion method for estimating joint angle parameters using the initialized pose matrix.
-   A hierarchical pipeline for hand pose estimation that combines the global pose orientation and finger articulations in a principled way.

##### Efficiently Creating 3D Training Data for Fine Hand Pose Estimation. [\[PDF\]](https://arxiv.org/pdf/1605.03389.pdf) [\[Project Page\]](https://cvarlab.icg.tugraz.at/projects/hand_detection/) [\[Code\]](https://github.com/moberweger/semi-auto-anno)
_Markus Oberweger, Gernot Riegler, Paul Wohlhart, Vincent Lepetit_
We  propose  a  semi-automated  method  for efficiently  and  accurately  labeling  each  frame  of  a  hand depth  video  with  the  corresponding  3D  locations  of  the joints:   The  user  is  asked  to  provide  only  an  estimate  of the 2D  reprojections of  the  visible  joints  in  some  reference frames, which are automatically selected to minimize the labeling work by efficiently optimizing a sub-modular loss  function.

##### Fits Like a Glove: Rapid and Reliable Hand Shape Personalization.  [\[PDF\]](http://www.samehkhamis.com/tan-cvpr2016.pdf) [\[Project Page\]](http://campar.in.tum.de/Main/DavidTan)
_David Joseph Tan, Thomas Cashman, Jonathan Taylor, Andrew Fitzgibbon, Daniel Tarlow, Sameh Khamis, Shahram Izadi, Jamie Shotton_
![Alt text](media/cvpr16-fits-like-glove.png?raw=true "Optional Title")
We minimize an energy based on a sum of render-and-compare cost functions called the _golden energy_.
However, this energy is only piecewise continuous, due to pixels crossing occlusion boundaries, and is therefore not obviously amenable to efficient gradient-based optimization.
A key insight is that the energy is the combination of a smooth low-frequency
function with a high-frequency, low-amplitude, piecewise-continuous function.
A central finite difference approximation with a suitable step size can therefore jump over the dis-continuities to obtain a good approximation to the energy’s
low-frequency behavior, allowing efficient gradient-based optimization.

### 2015 ICCV
##### Training a Feedback Loop for Hand Pose Estimation. [\[PDF\]](https://cvarlab.icg.tugraz.at/pubs/oberweger_iccv15.pdf) [\[Project Page\]](https://cvarlab.icg.tugraz.at/projects/hand_detection/)
_Markus Oberweger, Paul Wohlhart, Vincent Lepetit_

##### Opening the Black Box: Hierarchical Sampling Optimization for Estimating Human Hand Pose.  [\[PDF\]](http://www.iis.ee.ic.ac.uk/dtang/iccv_2015_camready.pdf)
_Danhang Tang, Jonathan Taylor, Pushmeet Kohli, Cem Keskin, Tae-Kyun Kim, Jamie Shotton_

##### Depth-based hand pose estimation: data, methods, and challenges. [\[PDF\]](https://arxiv.org/pdf/1504.06378) [\[Project Page\]](http://www.ics.uci.edu/~jsupanci/#HandData) [\[Code\]](https://github.com/jsupancic/deep_hand_pose)
_James Supancic III, Deva Ramanan, Gregory Rogez, Yi Yang, Jamie Shotton_

##### 3D Hand Pose Estimation Using Randomized Decision Forest with Segmentation Index Points. [\[PDF\]](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwipy5fi9OvSAhUWwGMKHdSqDzoQFggeMAA&url=http%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_iccv_2015%2Fpapers%2FLi_3D_Hand_Pose_ICCV_2015_paper.pdf&usg=AFQjCNGT2imZQPCrX5ggOGGDZoKmokLsAw&sig2=3U22HjWavqmtFM7eO550Fw)
_Peiyi Li, Haibin Ling_

##### A collaborative filtering approach to real-time hand pose estimation. [\[PDF\]](https://engineering.purdue.edu/cdesign/wp/wp-content/uploads/2015/08/iccv_2015_hand_pose_estimation.pdf) [\[Project Page\]](https://engineering.purdue.edu/cdesign/wp/a-collaborative-filtering-approach-to-real-time-hand-pose-estimation/)
_Chiho Choi, Ayan Sinha, Joon Hee Choi, Sujin Jang, Karthik Ramani_

##### Lending A Hand: Detecting Hands and Recognizing Activities in Complex Egocentric Interactions. [\[PDF\]](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjWqf689evSAhVQ4mMKHd_aBCoQFggbMAA&url=http%3A%2F%2Fvision.soic.indiana.edu%2Fpapers%2Fegohands2015iccv.pdf&usg=AFQjCNEpictJfsVL4DHKGE2tm5DMe3G7-w&sig2=LFRJs92qAWwQjAZS1VNMMA)
_Sven Bambach, Stefan Lee, David Crandall, Chen Yu_

##### Understanding Everyday Hands in Action from RGB-D Images. [\[PDF\]](https://www.google.com.sg/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwiowKvp9evSAhVJ5GMKHVk-B_gQFggbMAA&url=http%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_iccv_2015%2Fpapers%2FRogez_Understanding_Everyday_Hands_ICCV_2015_paper.pdf&usg=AFQjCNEIzPwbdJMme10hqQzSwy0rJNDhIQ&sig2=FI8vFqqQdoVmcrGodTJ7NQ)
_Gregory Rogez, James Supancic III, Deva Ramanan_

### 2015 CVPR
##### Cascaded Hand Pose Regression.  [\[PDF\]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Sun_Cascaded_Hand_Pose_2015_CVPR_paper.PDF)
_Xiao Sun, Yichen Wei, Shuang Liang, Xiaoou Tang, and Jian Sun_

##### Fast and Robust Hand Tracking Using Detection-Guided Optimization. [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/FastHandTracker/content/FastHandTracker_CVPR2015.pdf) [\[Project Page\]](http://handtracker.mpi-inf.mpg.de/projects/FastHandTracker/)
_Srinath Sridhar, Franziska Mueller, Antti Oulasvirta, Christian Theobalt_

##### Learning an Efficient Model of Hand Shape Variation from Depth Images. [\[PDF\]](http://www.samehkhamis.com/khamis-cvpr2015.pdf)
_Sameh Khamis, Jonathan Taylor, Jamie Shotton, Cem Keskin, Shahram Izadi, Andrew Fitzgibbon_

### 2015 Others
##### \[2015 CHI\] Accurate, Robust, and Flexible Real-time Hand Tracking. [\[PDF\]](http://www.cs.toronto.edu/~jtaylor/papers/CHI2015-HandTracking.pdf) [\[Project Page\]](https://www.microsoft.com/en-us/research/publication/accurate-robust-and-flexible-real-time-hand-tracking/)
_Toby Sharp, Cem Keskin, Duncan Robertson, Jonathan Taylor, Jamie Shotton, David Kim, Christoph Rhemann, Ido Leichter, Alon Vinnikov, Yichen Wei, Daniel Freedman, Pushmeet Kohli, Eyal Krupka, Andrew Fitzgibbon, Shahram Izadi_

##### \[2015 CVWW\]Hands Deep in Deep Learning for Hand Pose Estimation. [\[PDF\]](https://cvarlab.icg.tugraz.at/pubs/oberweger_cvww15.pdf) [\[Project Page\]](https://cvarlab.icg.tugraz.at/projects/hand_detection/) [\[Code\]](https://github.com/moberweger/deep-prior)
_Markus Oberweger, Paul Wohlhart, Vincent Lepetit_

##### \[2015 FG\]Combining Discriminative and Model Based Approaches for Hand Pose Estimation. [\[PDF\]](http://www.krejov.com/uploads/2/4/0/5/24053627/final_fg2015.pdf) [\[Project Page\]](http://www.krejov.com/hand-pose-estimation.html)
_Philip Krejov, Andrew Gilbert, Richard Bowden_

##### \[2015 SGP\] Robust Articulated-ICP for Real-Time Hand Tracking. [\[PDF\]](http://gfx.uvic.ca/pubs/2015/htrack//paper.pdf)  [\[Project Page\]](http://lgg.epfl.ch/publications/2015/Htrack_ICP/index.php) [\[Code\]](https://github.com/OpenGP/htrack)
_Anastasia Tkach, Mark Pauly, Andrea Tagliasacchi_

### 2014 CVPR
##### Realtime and robust hand tracking from depth. [\[PDF\]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/yichenw-cvpr14_handtracking.pdf) [\[Project Page\]](https://www.microsoft.com/en-us/research/people/yichenw/)
_Chen Qian, Xiao Sun, Yichen Wei, Xiaoou Tang and Jian Sun_

##### Latent regression forest: Structured estimation of 3d articulated hand posture. [\[PDF\]](http://www.iis.ee.ic.ac.uk/dtang/cvpr_14.pdf) [\[Project Page\]](http://www.iis.ee.ic.ac.uk/dtang/hand.html)
_Danhang Tang, Hyung Jin Chang, Alykhan Tejani, T-K. Kim_

##### User-specific hand modeling from monocular depth sequences. [\[PDF\]](http://www.cs.toronto.edu/~jtaylor/papers/CVPR2014-UserSpecificHandModeling.pdf) [\[Project Page\]](https://www.microsoft.com/en-us/research/publication/user-specific-hand-modeling-from-monocular-depth-sequences/)
_Jonathan Taylor, Richard Stebbing, Varun Ramakrishna, Cem Keskin, Jamie Shotton, Shahram Izadi, Aaron Hertzmann, Andrew Fitzgibbon_

##### Evolutionary Quasi-random Search for Hand Articulations Tracking. [\[PDF\]](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwj5wfPQsuzTAhUMxbwKHcnBBRUQFggnMAA&url=http%3A%2F%2Fwww.cv-foundation.org%2Fopenaccess%2Fcontent_cvpr_2014%2Fpapers%2FOikonomidis_Evolutionary_Quasi-random_Search_2014_CVPR_paper.pdf&usg=AFQjCNFPvY-vHE1GyUwxg8I0_R5OUj4QAA&sig2=ZsQ-rh6U2m0eijvJXQ817A) [\[Project Page\]](http://users.ics.forth.gr/~oikonom/pb/publications)
_Iason Oikonomidis, Manolis IA Lourakis, Antonis A Argyros_

### 2014 Others & Before
##### \[2014 SIGGRAPH\] Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks. [\[PDF\]](http://cims.nyu.edu/~tompson/others/TOG_2014_paper_PREPRINT.pdf) [\[Project Page\]](http://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)
_Jonathan Tompson, Murphy Stein, Yann Lecun and Ken Perlin_

##### \[2013 ICCV\] Real-time Articulated Hand Pose Estimation using Semi-supervised Transductive Regression Forests. [\[PDF\]](http://www.iis.ee.ic.ac.uk/dtang/iccv_13.pdf) [\[Project Page\]](http://www.iis.ee.ic.ac.uk/dtang/hand.html)
_Danhang Tang, Tsz Ho Yu and T-K. Kim_

##### \[2013 ICCV\] Interactive Markerless Articulated Hand Motion Tracking Using RGB and Depth Data. [\[PDF\]](http://handtracker.mpi-inf.mpg.de/projects/handtracker_iccv2013/content/handtracker_iccv2013.pdf) [\[Project Page\]](http://handtracker.mpi-inf.mpg.de/projects/handtracker_iccv2013/)
_Srinath Sridhar, Antti Oulasvirta, Christian Theobalt_

##### \[2013 ICCV\] Efficient Hand Pose Estimation from a Single Depth Image. [\[PDF\]](http://web.bii.a-star.edu.sg/~xuchi/pdf/iccv2013.pdf) [\[Project Page\]](http://web.bii.a-star.edu.sg/~xuchi/dhand.htm)
_Chi Xu, Li Cheng_

##### \[2012 ECCV\] Hand pose estimation and hand shape classification using multi-layered randomized decision forests
_Cem KeskinFurkan, KıraçYunus Emre, KaraLale Akarun_

##### \[2011 CVPRW\] Real Time Hand Pose Estimation using Depth Sensors. [\[PDF\]](http://www.cp.jku.at/teaching/praktika/imageproc/bodyparts_Algorithmus1.pdf)
_Cem Keskin, Furkan Kırac, Yunus Emre Kara, Lale Akarun_

##### \[2011 BMVC\] Efficient Model-based 3D Tracking of Hand Articulations using Kinect. [\[PDF\]](http://www.cp.jku.at/teaching/praktika/imageproc/bodyparts_Algorithmus1.pdf) [\[Project Page\]](http://users.ics.forth.gr/~argyros/research/kinecthandtracking.htm) [\[Code\]](https://github.com/FORTH-ModelBasedTracker/HandTracker)
_Iason Oikonomidis, Nikolaos Kyriazis, Antonis A. Argyros_

## Journal Papers

##### \[2016 IJCV\] Lie-X: Depth Image Based Articulated Object Pose Estimation, Tracking, and Action Recognition on Lie Groups. [\[PDF\]](http://web.bii.a-star.edu.sg/~xuchi/pdf/XuEtAl_IJCV16.pdf) [\[Project Page\]](http://web.bii.a-star.edu.sg/~xuchi/dhand.htm)
_Chi Xu, Lakshmi Narasimhan Govindarajan, Yu Zhang, Li Cheng_

##### \[2016 TPAMI\] Latent Regression Forest: Structured Estimation of 3D Hand Poses.
_Danhang Tang, Hyung Chang, Alykhan Tejani, Tae-Kyun Kim_

##### \[2016 CVIU\] Guided Optimisation through Classification and Regression for Hand Pose Estimation. [\[PDF\]](http://www.krejov.com/uploads/2/4/0/5/24053627/1-s2.0-s107731421630193x-main.pdf) [\[Project Page\]](http://www.krejov.com/hand-pose-estimation.html)
_Philip Krejov, Andrew Gilbert, Richard Bowden_

##### \[2015 TCSVT\] Resolving Ambiguous Hand Pose Predictions by Exploiting Part Correlations. [\[PDF\]](https://fae1051c-a-62cb3a1a-s-sites.googlegroups.com/site/seraphlh/2014TCSVT_HandPoseEstimation.pdf?attachauth=ANoY7cqF4PK7sqq9tp3b6n9qdhnx-6DqQwpjMKZIqnM8G-dMWwJFDDj35udChAet0y5jNOepL2MTujtVVwKui3rx8hogCKmYCZba_xEtjyMZII5MepMLrSNMYUOL7TGgkPGFHT7wvYR_dUIw_82Ok2MCo2rFwyTErNVmvlqkXuGNAaI8orzQzsKLfv1PiwVY32NWPlIz_oWuHL1M3slA97O-jXt511socyqDDj-azzhEodhzFjtz1BI%3D&attredirects=0)
_Hui Liang, Junsong Yuan, Daniel Thalmann_

##### \[2015 IJCV\] Estimate Hand Poses Efficiently from Single Depth Images. [\[PDF\]](https://web.bii.a-star.edu.sg/~xuchi/pdf/XuEtAl_IJCV15.pdf) [\[Project Page\]](http://web.bii.a-star.edu.sg/~xuchi/dhand.htm)  [\[Code\]](https://github.com/lzddzh/HandPoseEstimation)
_Chi Xu, Ashwin Nanjappa, Xiaowei Zhang, Li Cheng_

##### \[2014 TMM\] Parsing the Hand in Depth Images. [\[PDF\]](https://fae1051c-a-62cb3a1a-s-sites.googlegroups.com/site/seraphlh/attachments/2014TMM_HandParsing.pdf?attachauth=ANoY7crJCn_-tr0um1h8DhY3QtG8ngGn8jsllw1_S2ykaSsRGXvoeHWz7MW4DJ4KvQbXVd3nIsyWxEcs4rEn04TjtUaOTEMm7llUEP2e4renxgUj7G2DrVKDZzYg3Dbat1xhrvbz0BdjBoGrvxIniQLQ3Jyzs58UCDGSlzo-sGiOdmgMC072ZOCIR9STMP1FDpQzq3WV9fIGMUycXQRyWLja08ADLZOeV3d0eGKO1NoNH8pxN5pDD6M%3D&attredirects=0) [\[Project Page\]](https://sites.google.com/site/seraphlh/projects)  [\[Code\]](https://github.com/shrekei/RandomDecisionForest)
_Hui Liang, Junsong Yuan, Daniel Thalmann_

## Theses
##### \[2016 Thesis\] 3D hand pose regression with variants of decision forests. [\[PDF\]](http://people.mpi-inf.mpg.de/~ssridhar/pubs/Dissertation_SrinathSridhar.pdf)
_[Srinath Sridhar](http://people.mpi-inf.mpg.de/~ssridhar/),  Max Planck Institute for Informatics_

##### \[2016 Thesis\] 3D hand pose regression with variants of decision forests. [\[PDF\]](https://spiral.imperial.ac.uk/bitstream/10044/1/31531/1/Tang-D-2016-PhD-Thesis.pdf) [\[Project Page\]](https://spiral.imperial.ac.uk/handle/10044/1/31531)
_[Danhang Tang](http://www.iis.ee.ic.ac.uk/dtang/), Imperial College London_

##### \[2016 Thesis\] Deep Learning for Human Motion Analysis. [\[PDF\]](https://tel.archives-ouvertes.fr/tel-01470466v1/document) [\[Project Page\]](https://tel.archives-ouvertes.fr/tel-01470466v1)
_[Natalia Neverova](http://liris.cnrs.fr/natalia.neverova/), National Institut of Applied Science (INSA de Lyon), France_

##### \[2016 Thesis\] Real time hand pose estimation for human computer interaction. [\[PDF\]](http://epubs.surrey.ac.uk/809973/1/thesis.pdf) [\[Project Page\]](http://epubs.surrey.ac.uk/809973/)
_[Philip Krejov](http://www.krejov.com/), University of Surrey_

##### \[2015 Thesis\] Efficient Tracking of the 3D Articulated Motion of Human Hands. [\[PDF\]](http://users.ics.forth.gr/~oikonom/pb/oikonomidisPhDthesis.pdf)
_[Iason Oikonomidis](http://users.ics.forth.gr/~oikonom/pb/), University of Crete_

##### \[2015 Thesis\] Vision-based hand pose estimation and gesture recognition. [\[PDF\]](https://repository.ntu.edu.sg/bitstream/handle/10356/65842/ThesisMain.pdf?sequence=1&isAllowed=y)
_[Hui Liang](https://sites.google.com/site/seraphlh/home), Nanyang Technological University_

##### \[2015 Thesis\] Localization of Humans in Images Using Convolutional Networks. [\[PDF\]](http://www.cims.nyu.edu/~tompson/others/thesis.pdf)
_[Jonathan Tompson](http://cims.nyu.edu/~tompson/), New York University_

## Other Related Papers
##### \[2017 Neurocomputing\] Multi-task, Multi-domain Learning: application to semantic segmentation and pose regression.
_Fourure, Damien, et al._

##### [\[arXiv:1704.02463\]](https://arxiv.org/abs/1704.02463) First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations. [\[PDF\]](https://arxiv.org/pdf/1704.02463.pdf)
_Guillermo Garcia-Hernando, Shanxin Yuan, Seungryul Baek, Tae-Kyun Kim_

##### \[2017 CVPR\] SurfNet: Generating 3D shape surfaces using deep residual networks. [\[PDF\]](https://engineering.purdue.edu/cdesign/wp/wp-content/uploads/2017/03/Sinha_CVPR17.pdf)
_Ayan Sinha, Asim Unmesh, Qixing Huang, Karthik Ramani_

##### \[2017 CVPR\] Learning from Simulated and Unsupervised Images through Adversarial Training. [\[PDF\]](https://arxiv.org/pdf/1511.06728) [\[Project Page\]](https://machinelearning.apple.com/2017/07/07/GAN.html) [\[Code-Tensorflow\]](https://github.com/carpedm20/simulated-unsupervised-tensorflow) [\[Code-Keras\]](https://github.com/wayaai/SimGAN) [\[Code-Tensorflow-NYU-Hand\]](https://github.com/shinseung428/simGAN_NYU_Hand)
_Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Josh Susskind, Wenda Wang, Russ Webb_

##### \[2016 3DV\] Learning to Navigate the Energy Landscape. [\[PDF\]](http://www.robots.ox.ac.uk/~tvg/publications/2016/LNEL.pdf) [\[Project Page\]](http://graphics.stanford.edu/projects/reloc/)
_Julien Valentin, Angela Dai, Matthias Niessner, Pushmeet Kohli, Philip H.S. Torr, Shahram Izadi_

##### [2016 NIPS] DISCO Nets : Dissimilarity Coefficient Networks. [\[PDF\]](http://www.robots.ox.ac.uk/~diane/DISCONET_camera_ready.pdf) [\[Project Page\]](http://www.robots.ox.ac.uk/~diane/DiscoNets.html) [\[Code\]](http://www.robots.ox.ac.uk/~diane/DISCONETS.zip)
_Diane Bouchacourt, M. Pawan Kumar, Sebastian Nowozin_

---
\* indicates equal contribution
