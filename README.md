# Simple Neural Network
 ---
- This is using the classic MNIST handwritten digits dataset to train a neural network to recognize handwritten digits
- Made without the use of high level machine learning libraries such as TensorFlow and Pytorch
- Only used the numpy library to do linear algebra
- Model Structure:
  - Flatten -> Dense (x2) -> Softmax
  - The activation function is the Sigmoid Activation
  - Loss is calculated with the categorical cross-entropy loss function
    - By extension, the labels were converted to one-hot encoded vectors  

* Inspired by [this](https://www.youtube.com/watch?v=pauPCy_s0Ok&t=1320s&ab_channel=TheIndependentCode) video!
