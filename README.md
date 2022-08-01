# Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data

We provied pytorch(python) implementations of **Deep learning based on parameterized physical forward model for adaptive holographic imaging with unpaired data**. This code was written by **Chanseok Lee**.

# Overview
Holographic imaging poses the ill-posed inverse mapping problem of retrieving complex amplitude maps from measured diffraction intensity patterns. The existing deep learning methods for holographic imaging often solely depends on the statistical relation between the given data distributions, compromising its reliability in practical imaging configurations where physical perturbations exist in various forms, such as mechanical movement and optical fluctuation. Here, we present a deep learning method based a parameterized physical forward model that reconstructs both the complex amplitude and the range of objects under highly perturbative configurations where the object-to-sensor distance is set beyond the range of a given training data. To prove the reliability in practical biomedical applications, we demonstrate holographic imaging of red blood cells flowing in a cluster and diverse types of tissue sections presented without any ground truth data. Our results suggest that the proposed approach permits the adaptability of deep learning methods to deterministic perturbations, and therefore, extends their applicability to a wide range of inverse problems in imaging. 
<p align = "center">
<img src="/image/MainFigure_1.png" width="800" height="760">
</p>

# System Requirements
## Clone
```
git clone https://github.com/csleemooo/Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging.git
```

## Packages
The following libraries are necessary for running the codes.
- pytorch==1.11.0
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
- Windows 10
- CUDA 11.3
- cuDnn 8.2

# Train
We shared 300 patches of ground truth data and 100 patches of diffraction pattern intensity for each distance - 7, 9, 11, 13, 15, and 17mm of polystyrene bead data.
The data ca be downloaded from [here](https://drive.google.com/drive/u/2/folders/1kOjNYEzl8IAgIC-VBdlFQtojzrqdr2YU)
The data path should be

ex)
[DATA_PATH]/polystyrene_bead [DATA_NAME]
  - train
    - gt_amplitude
    - gt_phase
    - holography 
  - test
    - gt_amplitude
    - gt_phase
    - holography

[DATA_PATH_TEST]/polystyrene_bead_test [DATA_NAME]
  - train
    - gt_amplitude
    - gt_phase
    - holography 
  - test
    - gt_amplitude
    - gt_phase
    - holography

The proposed network can be trained with following command
```
python train_main.py --data_root [DATA_PATH] --data_name_gt [DATA_NAME] --data_name_diffraction [DATA_NAME] --data_name_test [DATA_NAME_TEST] --train_diffraction_list 13 --test_diffraction_list 7,9,11,13,15,17 --result_root [RESULT_PATH] --experiment [EXPERIMENT_NAME] --distance_min 7 --distance_max 17 --train_gt_ratio 1 --train_diffraction_ratio 1
```

# Reproduce
Here, user can reproduce the reported results from Fig 2 to Fig 5 by following instructions.  
Also, trained network parameters used in this study can be downloaded from [here](https://drive.google.com/drive/folders/1Y6R8plKylzHNT4wkBEA4GeOreY9id1xm?usp=sharing.). Download folders and put them to **./model_parameters** folder.  

### Demonstration of simultaneous reconstruction of complex amplitude and object distance
Here, simultaneous reconstruction of complex amplitude and distance range from single hologram intensity measurement is demonstrated. By integrating parameterized forward model with CycleGAN architecture, the proposed method can successfully reconstruct complex amplitude and object distance simultaneously. Run the following command.  
```
python result./result_fig2.py
```
Running time(CPU): 5s

### Demonstration of adaptive holographic imaging
The networks were trained in the situation where only single-dpeth hologram intensity measurements can be acquired. As the proposed model parameterized the physical degree of freedom (i.e. distance), the out-of-distribution data(i.e. hologram intensities measured at different distances) can be handled. Complex amplitude of polystyrene microsphere reconstructed from four different methods -U-Net, CycleGAN, PhaseGAN, and the proposed- and corresponding ground truth images are compared. Run the following command.  
```
python result./result_fig3.py
```
Running time(CPU): 15s

### Demonstration of holographic imaging of RBCs in a dynamic environment
To demonstrate the practicality of the proposed method, a series of hologram intensities of red blood cells that drift along the slide glass is measured and used as network input. 5 intermediate frames of input hologram intensities and the corresponding reconstructed complex amplitude are presented. Run the following command.
```
python result./result_fig4.py
```
Running time(CPU): 5s

### Holographic imaging of histology slides without ground truth
Here we assumed that acquiring paired data between complex amplitude and hologram intensity are not accessible. In this situation, complex amplitude of appendix and colon reconstructed from three different methods -CycleGAN, PhaseGAN, and the proposed- and corresponding ground truth images are compared. Notably, supervised method -U-Net- is omitted as paired data is not accessible. Run the following command.  
```
python result./result_fig5.py
```
Running time(CPU): 15s
