# Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data

We provied pytorch(python) implementations of **Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data**. This code was written by **Chanseok Lee**.
Last update: 2022.08.03

# Overview
Holographic imaging poses the ill-posed inverse mapping problem of retrieving complex amplitude maps from measured diffraction intensity patterns. The existing deep learning methods for holographic imaging often solely depends on the statistical relation between the given data distributions, compromising its reliability in practical imaging configurations where physical perturbations exist in various forms, such as mechanical movement and optical fluctuation. Here, we present a deep learning method based a parameterized physical forward model that reconstructs both the complex amplitude and the range of objects under highly perturbative configurations where the object-to-sensor distance is set beyond the range of a given training data. To prove the reliability in practical biomedical applications, we demonstrate holographic imaging of red blood cells flowing in a cluster and diverse types of tissue sections presented without any ground truth data. Our results suggest that the proposed approach permits the adaptability of deep learning methods to deterministic perturbations, and therefore, extends their applicability to a wide range of inverse problems in imaging. 
<p align = "center">
<img src="/image/MainFigure_1.png" width="800" height="760">
</p>

# network structure
<p align = "center">
<img src="/image/SupplementaryFigure_1.png" width="800" height="400">
</p>

# System Requirements
## Clone
```
git clone https://github.com/csleemooo/Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging.git
```
Move to the cloned directory
```
cd Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging
```
## Packages
The following libraries are necessary for running the codes.
- torch==1.11.0
- torchvision==0.12.0
- phasepack==1.3
- numpy==1.21.5
- Pillow==9.0.1
- matplotlib==3.5.2
- scipy==1.7.3
- opencv-python==4.6.0.66
- pyfftw==0.13.0

Please install requirements using below command.
```
pip install -r requirements.txt
```
which should install in about few minutes.

## Environements
The package development version is tested on windows. The developmental version of the package has been tested on the following systems and drivers.
- Windows 10/Linux/Mac
- CUDA 11.3
- cuDnn 8.2
- RTX3080 Ti

# Train
The **Adaptive holographic imaging** and **Holographic imaging of RBCs in a dynamic environment** can be implemented through the following descriptions.

**Adaptive holographic imaging**

We shared 300 patches of complex amplitude and 100 patches of diffraction pattern intensity measured at distances of 7-17mm with 2mm spacing.The whole polystyrene bead data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.21378744). Also, parts of shared data are located in ./dataset/polystyrene_bead folder. User can run demo using this data. 


**Holographic imaging of RBCs in a dynamic environment**

we shared 600 patches of complex amplitude and 600 patches of diffraction pattern intensity measured at 24mm. The whole red blood cell data can be downloaded from [here](https://doi.org/10.6084/m9.figshare.21378744). Also, parts of shared data are located in ./dataset/red_blood_cell folder. User can run demo using this data. Note that there are no paired data between complex amplitude and diffraction intensity of red blood cell as we acquired the data in a dynamic environment. Also, as we did not measure the distance of diffraction intensity of red blood cell (setting distance range [18, 30] is enough).


Download **train.zip and test.zip** for training. 


The data path should be

ex)

[DATA_PATH]/[DATA_NAME]/
  - train
    - gt_amplitude
    - gt_phase
    - holography 

[DATA_PATH]/[DATA_NAME_TEST]/
  - test
    - gt_amplitude
    - gt_phase
    - holography

The proposed network can be trained with the following command.

To run demo
```
python train_main.py --data_root ./dataset --data_name_gt polystyrene_bead --data_name_diffraction polystyrene_bead --data_name_test polystyrene_bead --train_diffraction_list 13 --test_diffraction_list 7,9,11,13,15,17 --result_root ./ --experiment DEMO --distance_min 7 --distance_max 17 --train_gt_ratio 1 --train_diffraction_ratio 1 --iterations 10 --chk_iter 1
python train_main.py --data_root ./dataset --data_name_gt red_blood_cell --data_name_diffraction red_blood_cell --data_name_test red_blood_cell --train_diffraction_list 1 --test_diffraction_list 1,2,3,4,5,6,7,8,9,10 --result_root ./ --experiment DEMO --distance_min 18 --distance_max 30 --train_gt_ratio 1 --train_diffraction_ratio 1 --iterations 10 --chk_iter 1
```
To train the model with full dataset, [DATA_PATH], [DATA_NAME], [DATA_NAME_TEST], [RESULT_PATH], and [EXPERIMENT_NAME] should be filled by user.
```
python train_main.py --data_root [DATA_PATH] --data_name_gt [DATA_NAME] --data_name_diffraction [DATA_NAME] --data_name_test [DATA_NAME_TEST] --train_diffraction_list 13 --test_diffraction_list 7,9,11,13,15,17 --result_root [RESULT_PATH] --experiment [EXPERIMENT_NAME] --distance_min 7 --distance_max 17 --train_gt_ratio 1 --train_diffraction_ratio 1 --iterations 20000
```
The trained network parameters will be saved at [RESULT_PATH]/[EXPERIMENT_NAME]. Also, the network parameters trained with full dataset that used in this study can be downloaded from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.).

Running time: up to 4 hours on the tested environment. 

# Test
We shared additional test data. Download **test_4fov.zip** from [here](https://doi.org/10.6084/m9.figshare.21378744). Make sure that the network parameters are saved somewhere before run the test code.

The data path should be

ex)

[DATA_PATH_TEST]/
  - fov1
    - test
  - fov2
    - test
  ...

The network can be tested with the following command.

To run demo, user should run the train demo code first. If not, the network parameters should be downloaded from [here](https://drive.google.com/file/d/1Ov276n377Siad4NbgLAwZqtvt4soi-4U/view?usp=sharing) and saved in ./DEMO folder.
```
python test_main.py --data_root ./dataset --data_name_test polystyrene_bead --model_root ./ --experiment DEMO --test_diffraction_list 7,8,9,10,11,12,13,14,15,16,17
python test_main.py --data_root ./dataset --data_name_test red_blood_cell --model_root ./ --experiment DEMO --test_diffraction_list 1,2,3,4,5,6,7,8,9,10
```
[FOV]: fov1, fov2, fov3, fov4
```
python test_main.py --data_root [DATA_PATH_TEST] --data_name_test [FOV] --model_root [RESULT_PATH] --experiment [EXPERIMENT_NAME] --test_diffraction_list 7,8,9,10,11,12,13,14,15,16,17
```
Test result will be saved at [RESULT_PATH]/[EXPERIMENT_NAME]/test_image_result.

Running time: less than a minute on the tested environment. 

# Reproduce
Here, user can reproduce the reported results from Fig 2 to Fig 5 by following instructions.  
Also, trained network parameters used in this study can be downloaded from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.). Download folders and put them to **./model_parameters** folder.  

### Demonstration of simultaneous reconstruction of complex amplitude and object distance
Here, simultaneous reconstruction of complex amplitude and distance range from a single diffraction intensity measurement is demonstrated. By integrating parameterized forward model with CycleGAN architecture, the proposed method can successfully reconstruct complex amplitude and object distance simultaneously. Run the following command.  
```
python result./result_fig2.py
```
Running time(CPU): 5s

### Demonstration of adaptive holographic imaging
The networks were trained in the situation where only single-dpeth diffraction intensity measurements can be acquired. As the proposed model parameterized the physical degree of freedom (i.e. distance), the out-of-distribution data(i.e. diffraction intensities measured at different distances) can be handled. Complex amplitude of polystyrene microsphere reconstructed from four different methods -U-Net, CycleGAN, PhaseGAN, and the proposed- and corresponding ground truth images are compared. Run the following command.  
```
python result./result_fig3.py
```
Running time(CPU): 15s

### Demonstration of holographic imaging of RBCs in a dynamic environment
To demonstrate the practicality of the proposed method, a series of diffraction intensities of red blood cells that drift along the slide glass is measured and used as network input. 5 intermediate frames of input diffraction intensities and the corresponding reconstructed complex amplitude are presented. Run the following command.
```
python result./result_fig4.py
```
Running time(CPU): 5s

### Holographic imaging of histology slides without ground truth
Here we assumed that acquiring paired data between complex amplitude and diffraction intensity are not accessible. In this situation, complex amplitude of appendix and colon reconstructed from three different methods -CycleGAN, PhaseGAN, and the proposed- and corresponding ground truth images are compared. Notably, supervised method -U-Net- is omitted as paired data is not accessible. Run the following command.  
```
python result./result_fig5.py
```
Running time(CPU): 15s
