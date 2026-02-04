Landmark-Guided CNN for Facial Emotion Recognition (FER-2013)

This repository contains the implementation of a Landmark-Guided Convolutional Neural Network (LM-CNN) for facial emotion recognition using the FER-2013 dataset. The project re-implements a baseline CNN (Proposed Model-2) and extends it by incorporating facial landmark heatmaps as an additional input channel, enabling the network to jointly learn appearance-based and geometry-based facial features.

Project Overview
Facial expression recognition systems based only on image appearance often ignore the underlying facial geometry. This project introduces a landmark-guided learning strategy in which facial landmark heatmaps are fused with grayscale facial images to provide structural guidance to the CNN.

Two models are implemented and compared:
Baseline CNN (Proposed Model-2)
Landmark-Guided CNN (LM-CNN)
Both models are evaluated on the FER-2013 dataset using 2-fold cross-validation.


