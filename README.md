Deep Learning Engine from Scratch
(NumPy)
A lightweight, high-performance deep learning engine built entirely from scratch using NumPy.
This project bypasses high-level frameworks like PyTorch or TensorFlow to implement core
neural network mechanics—including matrix calculus, backpropagation, activation derivatives,
and adaptive optimization—using raw matrix mathematics.
The engine features an advanced Multi-Task Learning (MTL) architecture capable of outputting
simultaneous classification and regression predictions from a shared hidden representation.

�� Features
 Pure NumPy Implementation: All forward and backward passes are calculated using optimized
matrix operations (np.dot, transpose manipulation).
 Multi-Task Learning Architecture: Features a split-head output layer that optimizes a joint loss
function for concurrent categorical classification and continuous regression tasks.
 Custom Activation &amp; Derivative Layers: Manual implementations of ReLU (hidden layers),
Sigmoid &amp; Softmax (output layers), featuring exact analytical gradients for stable
backpropagation.
 Scratch Adam Optimizer: A native implementation of the Adam (Adaptive Moment Estimation)
optimization algorithm, featuring first and second-moment tracking, bias correction, and decay.
 Evaluation Suite: Custom performance metrics including vector-mapped Mean Squared Error
(MSE), Categorical Cross-Entropy, and exact tracking using np.argmax.

��️ Project Structure
├── src/
│   ├── __pycache__/
│   ├── __init__.py      # Marks src as a Python package
│   ├── activations.py   # ReLU, Sigmoid, Softmax, and their derivatives
│   ├── datasets.py      # Data loading and synthetic generation utils
│   ├── layers.py        # Dense layers and forward/backward mechanics
│   ├── loss.py          # Multi-task loss functions (MSE + Cross-Entropy)
│   └── optimizers.py    # Custom Adam and SGD optimization math
├── .gitignore           # Ignores __pycache__ and system files
├── brain.pkl            # Serialized trained neural network weights/parameters
├── main.py              # Main training script logic and engine orchestration
├── spiral_result.png    # Visualized decision boundary result (e.g., spiral dataset)
└── usebrain.py          # Inference script to load brain.pkl and predict

⚙️ Installation &amp; Setup
Prerequisites: Ensure you have Python 3.8+ installed on your system.

1. Clone the Repository
git clone https://github.com/luckylittleman/numpy-net.git
cd numpy-net

2. Set Up a Virtual Environment (Recommended)
On Linux/macOS:
python3 -m venv venv
source venv/bin/activate

On Windows:
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
This project is built to prove core principles with minimal external abstraction, meaning NumPy is the
only production dependency.
pip install -r requirements.txt

(If a requirements.txt file isn&#39;t present, simply run `pip install numpy`)
�� Quick Start
To see the multi-task engine in action, you can run the synthetic training pipeline included in the
examples folder:
python examples/train_mtl.py

Basic API Usage Example
import numpy as np
from src.layers import DenseLayer
from src.activations import ReLU, Sigmoid
from src.optimizers import Adam

# Initialize a simple layer: 10 input features, 16 hidden neurons
layer1 = DenseLayer(input_dim=10, output_dim=16)
activation1 = ReLU()
# Forward Pass
X = np.random.randn(32, 10) # Batch of 32 samples
Z1 = layer1.forward(X)
A1 = activation1.forward(Z1)
# Backward Pass (Propagating a dummy gradient)
dA1 = np.random.randn(32, 16)
dZ1 = activation1.backward(dA1)
dX = layer1.backward(dZ1)
# Update parameters using our custom Adam optimizer
optimizer = Adam(learning_rate=0.001)
layer1.weights = optimizer.update(layer1.weights, layer1.d_weights)
layer1.biases = optimizer.update(layer1.biases, layer1.d_biases)

�� Contribution Instructions
Contributions are welcome! Whether you are optimizing a matrix operation, fixing an analytical
gradient bug, or adding new activation layers, follow the standards listed below.
Code of Conduct &amp; Standards
 Mathematical Clarity: Keep variable names aligned with clear mathematical notation (e.g., dW
for weight gradients, dZ for pre-activation gradients).
 No Heavy Frameworks: Do not import or rely on PyTorch, TensorFlow, JAX, or Scikit-Learn in
the core implementation layer.
 Vectorization over Loops: Always prefer vectorized NumPy operations (np.dot, broadcasting)
over explicit Python for loops to ensure computational efficiency.
Contribution Workflow
 Fork the repository on GitHub.
 Create a Feature Branch off the main branch:
`git checkout -b feature/amazing-new-layer`
 Commit your changes with descriptive messages:
`git commit -m &quot;feat: add LeakyReLU layer and its backward derivative&quot;`
 Push your branch to GitHub:
`git push origin feature/amazing-new-layer`
 Open a Pull Request describing your changes, your mathematical reasoning, and verification
tests.

�� License
This project is open-source and available under the MIT License
