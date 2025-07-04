# CNN Pipeline Visualization - Convolution, Pooling & Flattening

An interactive 3D visualization tool for teaching Convolutional Neural Networks (CNNs), demonstrating the complete pipeline from RGB input through convolution, max pooling, and flattening operations.

## Features

### 🎯 Interactive 3D Visualization
- **8×8×3 RGB Input**: Three distinct color layers representing an RGB image
  - Red layer (front)
  - Green layer (middle)  
  - Blue layer (back)
- **Configurable Kernels**: Choose 1-8 kernels to see how many feature maps are generated
- **8×8 Feature Maps**: Multiple grayscale output layers showing convolution results (same size due to padding)
- **Max Pooling**: 2×2 max pooling with stride 2 creates 4×4 pooled feature maps
- **Flattening**: All pooled feature maps are flattened into a single column vector
- **Real-time Updates**: Interactive controls to adjust the number of kernels

### 🎮 User Controls
- **Kernel Count**: Slider to choose how many kernels (1-8) to visualize
- **Auto Rotation**: Automated rotation to view the complete pipeline
- **Reset View**: Return to optimal viewing angle
- **3D Navigation**: Mouse controls for rotating, zooming, and panning the view

### 📚 Educational Benefits
- Visualizes the complete CNN pipeline from input to flattened output
- Demonstrates how convolution creates feature maps
- Shows how max pooling reduces spatial dimensions while preserving features
- Illustrates how flattening prepares data for fully connected layers
- Highlights the dimensionality changes at each step of the CNN pipeline
- Perfect for explaining the complete convolutional section of CNN architectures

## Usage

1. Open `index.html` in a modern web browser
2. Use the control panel to:
   - Adjust the number of kernels (1-8) to see different pipeline configurations
   - Enable "Auto Rotate" to see the complete pipeline from all angles
   - Use "Reset View" to return to the optimal viewing position
3. Use mouse to rotate and zoom the 3D scene
4. Observe how changing the number of kernels affects:
   - The number of feature maps (8×8)
   - The number of pooled maps (4×4)
   - The size of the flattened vector

## Architecture Flow

```
Input RGB (8×8×3) → Convolution → Feature Maps (8×8×N) → Max Pool → Pooled Maps (4×4×N) → Flatten → Vector (16×N)
```

Where N = number of kernels (1-8)

## Technical Implementation

- **Three.js**: 3D graphics rendering and scene management
- **OrbitControls**: Interactive camera controls
- **WebGL**: Hardware-accelerated 3D rendering
- **Responsive Design**: Adapts to different screen sizes

## Educational Context

This visualization helps students understand:
- How RGB images are processed in CNNs as 3D tensors
- The concept of multiple kernels creating multiple feature maps
- How padding preserves spatial dimensions in convolution
- How max pooling reduces spatial dimensions while preserving important features
- How flattening converts 2D feature maps to 1D vectors for fully connected layers
- The complete data flow through the convolutional section of a CNN

Perfect for computer vision courses, deep learning workshops, and CNN tutorials!

## Browser Requirements

- Modern browser with WebGL support
- Recommended: Chrome, Firefox, Safari, or Edge (latest versions)
