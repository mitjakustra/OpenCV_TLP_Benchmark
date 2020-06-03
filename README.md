# OpenCV_TLP_Benchmark
This repository contains the code to analyse the tracking algoritms contained in OpenCV, according to the TLP benchmark. The TLP benchmark is a way of testing and evaluating object tracking algorithms, focused on long term tracking analysis. This aim of this project is the evaluation of the OpenCV tracking algorithms, present in version 4.3.0, with the exclusion of the GOTURN algorithm. 

## Requirements

To run the code you will need to install Python 3.5/3.6/3.7/3.8, as well as having the next libraries installed:
- Numpy
- OpenCV
- Matplotlib

## Usage

The code is divided into three separate .py files: 
  - Testiranje_algoritmov.py
  - Primerjava_algoritmov.py
  - Prikaz_algoritmov.py
The files should be run in the presented sequence.

This is to be paired with a few selected sequences from the TinyTLP dataset, available here: https://amoudgl.github.io/tlp/

The sequences are:
  - Alladin
  - Aquarium2
  - Badminton1
  - CarChase3
  - DriftCar1
  - ISS
  - Jet4
  - KinBall2
  - PolarBear1
 
The OpenCV object tracking algorithms (TRACKERTYPE) evaluated in this project are:
  - Boosting
  - MIL
  - KCF
  - TLD
  - MEDIANFLOW
  - MOSSE
  - CSRT
  
 
The following python programs should be run from the index folder, where the sequences and the "groundtruth_rect.txt" and "_score.txt" and "_test_result.txt" files are located inside the data folder, as shown below.
 
 ![](images/File_Directory_Diagram.png)
 
### Testiranje_algoritmov.py

  - Enter the name of the sequence,
  - The program then imports the selected sequence's groundtruth_rect.txt file,
  - The program creates a new (TRACKERTYPE)_test_result.txt file in the same directory as the groundtruth_rect.txt,
  - The program uses the bounding box of the first image of the sequence from the groundtruth_rect.txt file as the starting bounding box     for the tracking algorithm, so that all algorithms start from the same point. (Manual selection can be enabled with                     uncommenting of the code),
  - The program then initialises the first tracking algorithm, using the bounding box, starting the tracking process and shows the video     with the two bounding boxes,
  - During this the program writes lines to the (TRACKERTYPE)_test_result.txt, in the format: FRAME_NUM, X_COR, Y_COR, WIDTH, HEIGHT,       LOST,
  - The tracking algorithm runs its course until there are no more images left in the sequence.
  
 ### Primerjava_algoritmov.py
 
  - The program first to choose the sequence to benchmark,
  - Next, the program interates through all the tracking algorithms, completing the steps below:
  - For each sequence (video) and for each tracking algorithm, the program imports the (TRACKERTYPE)_test_result.txt file and the           groundtruth_rect.txt files and creates a new (TRACKERTYPE)_score.txt file, which is used to store the success, precision, LSM           metrics and its respective scores,
  - The program reads line by line both the (TRACKERTYPE)_test_result.txt file and the groundtruth_rest.txt file, which correspond to       the data from each frame from the chosen sequence,
  - The program calculates the success, precision and LSM metrics and appends them to the appropriate lists,
  - The program then writes the calculated values and scores into the new (TRACKERTYPE)_score.txt for the current tracker and the chosen     sequence in a chosen format.
  
  ### Prikaz_algoritmov.py

  - The program iterates through the trackers,
  - The program then iterates through all sequences,
  - For the current tracker and the current sequence, the appropriate (TRACKERTYPE)_score.txt is imported,
  - The program reads the data from the imported files and it appends them to the appropriate lists,
  - The program then calculates the average values of all sequences for the current tracker,
  - Lastly, the program shows the plots for the success, precision, LSM metrics, their appropriate scores and the attribute based           success plots.

The complete data containing the sequences and all .txt files is available here: https://drive.google.com/uc?id=1RvDSuuAwpsxY2-oMSJLnu49_M_47-X4i&export=download

The code and the comments inside the code are in Slovenian, due to this being a university project for the Faculty of Electrical Engineering, University of Ljubljana.

## References

1. Moudgil A., Gandhi V. Long-term Visual Object Tracking Benchmark, 2019
