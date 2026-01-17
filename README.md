# Multimodal DeepFake Detection using Spatiotemporal and Contrastive Supervision
## INTRODUCTION

Deepfakes have improved significantly because of breakthroughs in deep learning, making it feasible to produce very realistic but manipulated video or image contents. Such fake contents of media seriously threaten several domains, including misinformation, identity deception, digital security, and social trust. Conventional deep fake detection solutions based on single-modal or static visual characteristics would fail to generalize against highly sophisticated methods of attacks.

With respect to this issue, to enhance the effectiveness of detecting subtle manipulation patterns, this proposed work aims to adopt a unified framework for deepfake detection based on spatiotemporal and contrastive learning supervision.
In this approach, for effective deepfake image and video manipulation pattern detection, the proposed unified framework combines spatial learning to effectively focus on texture and spatial artifacts of human faces, along with temporal learning to focus on motion inconsistencies between images of video frames.

Further, the use of contrastive supervision is made to improve the ability of the discriminative model by maximizing the separation of the feature space of the real and generated samples. The learning technique allows the model to pay attention to significant differences instead of focusing on trivial patterns, hence improving generalization capability. The proposed method is expected to offer a more accurate, reliable, and efficient way of detecting deepfakes.

### Datasets used:
The datasets used for training this model are: FakeAVCeleb, Faceforensics++ and ASV Spoof 2019. The datasets are cleaned and a metadata CSV is created for tracking video path, split and manipulation.

## Spatiotemporal model- EfficientNet-B2 and bi-LSTM:

Starting with the spatiotemporal model, In our approach, we train the spatial model (EfficientNet-B2) on 1 random frame from each videos in the dataset.
Due to large class imbalance between real and fake videos in the dataset, three techniques of balanced sampling was implemented: class balanced sampling, Identity balanced sampling and Manipulation aware sampling.
Spatial heatmap to predict fakes:
<img width="4500" height="1500" alt="annotated_heatmaps" src="https://github.com/user-attachments/assets/2dbb83aa-2961-41bf-a482-88c5b2051319" />

# Spatial model predicting fake videos:
<img width="869" height="802" alt="true pos" src="https://github.com/user-attachments/assets/78b9a0da-6859-4298-8996-04d43054aa07" />

For temporal training, we performed feature training to extract .npy files from the dataset using the freezed backbone of the EfficientNet-B2 for faster and efficient training. Three types of temporal models were trained on the extracted features - Temporal Convolution Network (TCN), Bi-LSTM(Long Short Term Memory) and Temporal Transformer. The performance of each model as a comparison is plotted below.

<img width="1317" height="370" alt="Screenshot (99)" src="https://github.com/user-attachments/assets/d1b7ff3d-67a0-46ca-a314-f559eb7b91c8" />
<img width="1079" height="511" alt="Screenshot (98)" src="https://github.com/user-attachments/assets/ca378f1b-ce70-48d4-9584-2e75f23a51c5" />
The chosen bi-LSTM model is further fine-tuned using contrastive learning using Triplet Loss.

## Audio Training (CRNN):
For detecting audio inconsistencies, temporal audio models like CRNN is used on the widely recognized ASV Spoof 2019 dataset =, achieving an accuracy and AUC of 99%.

Audio comparison (Real vs Fake):
<img width="4552" height="2445" alt="fakeavceleb_spectrogram_comparison" src="https://github.com/user-attachments/assets/92ddc483-0d5d-4206-830b-3f5662eddbd2" />

