// CNN Convolution Step Visualization
class CNNVisualization {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Image layers properties
        this.imageLayers = []; // Array of 3 input layers (red, green, blue)
        this.outputLayers = []; // Array of output layers (feature maps)
        this.pooledLayers = []; // Array of pooled layers (max pooled feature maps)
        this.flattenedLayer = []; // Single column of flattened cubes
        this.fcInputLayer = []; // Fully connected input nodes (spheres)
        this.hiddenLayer = []; // Hidden layer nodes
        this.outputLayer = []; // Output layer node (single)
        this.connectionLines = []; // Lines connecting layers
        this.allCubes = []; // All cubes across input layers
        this.allOutputCubes = []; // All cubes in output layers
        this.allPooledCubes = []; // All cubes in pooled layers
        this.allFlattenedCubes = []; // All cubes in flattened layer
        this.allFCInputNodes = []; // All FC input spheres
        this.allHiddenNodes = []; // All hidden layer nodes
        this.allOutputNodes = []; // All output layer nodes
        this.originalMaterials = []; // Store original materials for input layers
        this.originalOutputMaterials = []; // Store original materials for output layers
        this.originalPooledMaterials = []; // Store original materials for pooled layers
        this.originalFlattenedMaterials = []; // Store original materials for flattened layer
        this.originalFCInputMaterials = []; // Store original materials for FC input layer
        this.originalHiddenMaterials = []; // Store original materials for hidden nodes
        this.originalOutputMaterials = []; // Store original materials for output nodes
        this.cubeSize = 0.8;
        this.spacing = 1.0;
        this.layerSize = 8; // 8x8 per input layer
        this.layerSpacing = 3.0; // Distance between input layers
        this.outputSize = 8; // 8x8 output (same as input due to padding)
        this.pooledSize = 4; // 4x4 pooled output (8/2 with stride 2)
        this.flattenedSize = 0; // Will be calculated based on number of kernels (4*4*numKernels)
        this.fcInputSize = 8; // Number of FC input nodes (configurable)
        this.hiddenSize = 4; // Number of hidden layer nodes
        this.fcOutputSize = 1; // Single output node for final output layer
        
        // Configuration
        this.numKernels = 3; // Number of kernels/output feature maps
        
        // Animation
        this.autoRotate = false;
        this.rotationSpeed = 0.01;
        
        this.init();
        this.setupControls();
        this.animate();
    }
    
    init() {
        // Create scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);
        
        // Create camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(38, 15, 20); // Position relative to center point
        
        // Create renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = false;
        document.getElementById('container').appendChild(this.renderer.domElement);
        
        // Create controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxDistance = 50;
        this.controls.minDistance = 5;
        
        // Set the target to the center of the network pipeline
        this.controls.target.set(23, 0, 0); // Center point between input (0) and output (46)
        
        // Add lights
        this.setupLights();
        
        // Create the visualization
        this.createImageLayers();
        this.createOutputLayers();
        this.createPooledLayers();
        this.createFlattenedLayer();
        this.createFCInputLayer();
        this.createHiddenLayer();
        this.createOutputNode();
        this.createConnections();
        this.updateExplanationText();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    setupLights() {
        // High ambient light for uniform illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
        this.scene.add(ambientLight);
        
        // Subtle directional light for depth
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.2);
        directionalLight.position.set(1, 1, 1);
        directionalLight.castShadow = false;
        this.scene.add(directionalLight);
    }
    
    createImageLayers() {
        // Clear existing layers
        this.imageLayers.forEach(layer => {
            layer.forEach(cube => {
                if (cube.parent) {
                    cube.parent.remove(cube);
                }
            });
        });
        this.imageLayers = [];
        this.allCubes = [];
        this.originalMaterials = [];
        
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        
        // Colors for each layer
        const layerColors = [
            0xff4444, // Red layer
            0x44ff44, // Green layer  
            0x4444ff  // Blue layer
        ];
        
        // Create 3 layers, each 8x8
        for (let layer = 0; layer < 3; layer++) {
            const layerGroup = new THREE.Group();
            const layerCubes = [];
            
            const material = new THREE.MeshLambertMaterial({
                color: layerColors[layer],
                transparent: false,
                opacity: 1.0
            });
            
            // Create 8x8 grid for this layer
            for (let x = 0; x < this.layerSize; x++) {
                for (let y = 0; y < this.layerSize; y++) {
                    const cube = new THREE.Mesh(geometry, material.clone());
                    
                    // Add wireframe edges
                    const edges = new THREE.EdgesGeometry(geometry);
                    const edgeMaterial = new THREE.LineBasicMaterial({ 
                        color: 0x000000,
                        linewidth: 2 
                    });
                    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                    cube.add(wireframe);
                    
                    // Position the cube
                    cube.position.set(
                        (x - 3.5) * this.spacing,
                        (y - 3.5) * this.spacing,
                        0
                    );
                    
                    // Store references
                    cube.layerIndex = layer;
                    cube.gridX = x;
                    cube.gridY = y;
                    
                    layerGroup.add(cube);
                    layerCubes.push(cube);
                    this.allCubes.push(cube);
                    this.originalMaterials.push(cube.material.clone());
                }
            }
            
            // Position the entire layer
            layerGroup.position.z = (layer - 1) * this.layerSpacing;
            
            this.imageLayers.push(layerCubes);
            this.scene.add(layerGroup);
        }
    }
    
    createOutputLayers() {
        this.clearOutputLayers();
        
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        
        // Gray material for output layers
        const material = new THREE.MeshLambertMaterial({
            color: 0x888888,
            transparent: false,
            opacity: 1.0
        });
        
        // Create output layers based on number of kernels
        for (let kernel = 0; kernel < this.numKernels; kernel++) {
            const outputGroup = new THREE.Group();
            const outputCubes = [];
            
            // Create 8x8 grid for this output layer (same size due to padding)
            for (let x = 0; x < this.outputSize; x++) {
                for (let y = 0; y < this.outputSize; y++) {
                    const cube = new THREE.Mesh(geometry, material.clone());
                    
                    // Add wireframe edges
                    const edges = new THREE.EdgesGeometry(geometry);
                    const edgeMaterial = new THREE.LineBasicMaterial({ 
                        color: 0x000000,
                        linewidth: 2 
                    });
                    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                    cube.add(wireframe);
                    
                    // Position the cube
                    cube.position.set(
                        (x - 3.5) * this.spacing,
                        (y - 3.5) * this.spacing,
                        0
                    );
                    
                    // Store references
                    cube.outputX = x;
                    cube.outputY = y;
                    cube.kernelIndex = kernel;
                    
                    outputGroup.add(cube);
                    outputCubes.push(cube);
                    this.allOutputCubes.push(cube);
                    this.originalOutputMaterials.push(cube.material.clone());
                }
            }
            
            // Position each output layer with spacing
            const outputX = 12; // Distance from input layers
            const outputZ = (kernel - (this.numKernels - 1) / 2) * 4; // Spread them out in Z
            outputGroup.position.set(outputX, 0, outputZ);
            outputGroup.userData = { isOutputGroup: true }; // Mark for removal
            
            this.outputLayers.push(outputCubes);
            this.scene.add(outputGroup);
        }
    }
    
    clearOutputLayers() {
        // Remove existing output layers from scene
        this.outputLayers.forEach(layer => {
            layer.forEach(cube => {
                if (cube.parent) {
                    cube.parent.remove(cube);
                }
            });
        });
        
        // Also remove the group containers from the scene
        this.scene.children = this.scene.children.filter(child => {
            if (child.userData && child.userData.isOutputGroup) {
                return false;
            }
            return true;
        });
        
        // Clear arrays
        this.outputLayers = [];
        this.allOutputCubes = [];
        this.originalOutputMaterials = [];
    }
    
    createPooledLayers() {
        this.clearPooledLayers();
        
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        
        // Darker gray material for pooled layers
        const material = new THREE.MeshLambertMaterial({
            color: 0x666666,
            transparent: false,
            opacity: 1.0
        });
        
        // Create pooled layers based on number of kernels
        for (let kernel = 0; kernel < this.numKernels; kernel++) {
            const pooledGroup = new THREE.Group();
            const pooledCubes = [];
            
            // Create 4x4 grid for this pooled layer (2x2 max pooling with stride 2)
            for (let x = 0; x < this.pooledSize; x++) {
                for (let y = 0; y < this.pooledSize; y++) {
                    const cube = new THREE.Mesh(geometry, material.clone());
                    
                    // Add wireframe edges
                    const edges = new THREE.EdgesGeometry(geometry);
                    const edgeMaterial = new THREE.LineBasicMaterial({ 
                        color: 0x000000,
                        linewidth: 2 
                    });
                    const wireframe = new THREE.LineSegments(edges, edgeMaterial);
                    cube.add(wireframe);
                    
                    // Position the cube
                    cube.position.set(
                        (x - 1.5) * this.spacing, // Center the 4x4 grid
                        (y - 1.5) * this.spacing,
                        0
                    );
                    
                    // Store references
                    cube.pooledX = x;
                    cube.pooledY = y;
                    cube.kernelIndex = kernel;
                    
                    pooledGroup.add(cube);
                    pooledCubes.push(cube);
                    this.allPooledCubes.push(cube);
                    this.originalPooledMaterials.push(cube.material.clone());
                }
            }
            
            // Position each pooled layer with spacing
            const pooledX = 20; // Distance from output layers
            const pooledZ = (kernel - (this.numKernels - 1) / 2) * 4; // Spread them out in Z
            pooledGroup.position.set(pooledX, 0, pooledZ);
            
            this.pooledLayers.push(pooledCubes);
            this.scene.add(pooledGroup);
        }
    }
    
    clearPooledLayers() {
        // Remove existing pooled layers from scene
        this.pooledLayers.forEach(layer => {
            layer.forEach(cube => {
                if (cube.parent) {
                    cube.parent.remove(cube);
                }
            });
        });
        
        // Clear arrays
        this.pooledLayers = [];
        this.allPooledCubes = [];
        this.originalPooledMaterials = [];
    }
    
    createFlattenedLayer() {
        this.clearFlattenedLayer();
        
        // Calculate flattened size: 4x4 pixels per pooled map * number of kernels
        this.flattenedSize = this.pooledSize * this.pooledSize * this.numKernels;
        
        const geometry = new THREE.BoxGeometry(this.cubeSize, this.cubeSize, this.cubeSize);
        
        // Purple material for flattened layer to distinguish it
        const material = new THREE.MeshLambertMaterial({
            color: 0x9966cc,
            transparent: false,
            opacity: 1.0
        });
        
        const flattenedGroup = new THREE.Group();
        
        // Create a single column of cubes
        for (let i = 0; i < this.flattenedSize; i++) {
            const cube = new THREE.Mesh(geometry, material.clone());
            
            // Add wireframe edges
            const edges = new THREE.EdgesGeometry(geometry);
            const edgeMaterial = new THREE.LineBasicMaterial({ 
                color: 0x000000,
                linewidth: 2 
            });
            const wireframe = new THREE.LineSegments(edges, edgeMaterial);
            cube.add(wireframe);
            
            // Position cubes in a vertical column with spacing
            cube.position.set(
                0, // Single column (x = 0)
                (i - (this.flattenedSize - 1) / 2) * this.spacing * 1.2, // Increased vertical spacing
                0
            );
            
            // Store references - calculate which pooled map and position this represents
            const kernelIndex = Math.floor(i / (this.pooledSize * this.pooledSize));
            const posInKernel = i % (this.pooledSize * this.pooledSize);
            const pooledX = Math.floor(posInKernel / this.pooledSize);
            const pooledY = posInKernel % this.pooledSize;
            
            cube.flattenedIndex = i;
            cube.sourceKernel = kernelIndex;
            cube.sourcePooledX = pooledX;
            cube.sourcePooledY = pooledY;
            
            flattenedGroup.add(cube);
            this.flattenedLayer.push(cube);
            this.allFlattenedCubes.push(cube);
            this.originalFlattenedMaterials.push(cube.material.clone());
        }
        
        // Position the flattened layer
        const flattenedX = 28; // Distance from pooled layers
        flattenedGroup.position.set(flattenedX, 0, 0);
        
        this.scene.add(flattenedGroup);
    }
    
    clearFlattenedLayer() {
        // Remove existing flattened layer from scene
        this.flattenedLayer.forEach(cube => {
            if (cube.parent) {
                cube.parent.remove(cube);
            }
        });
        
        // Clear arrays
        this.flattenedLayer = [];
        this.allFlattenedCubes = [];
        this.originalFlattenedMaterials = [];
    }
    
    createFCInputLayer() {
        this.clearFCInputLayer();
        
        // FC input size matches flattened size for 1:1 mapping
        this.fcInputSize = this.flattenedSize;
        
        const sphereGeometry = new THREE.SphereGeometry(0.3, 16, 12); // Smaller spheres
        
        // Orange material for FC input nodes to distinguish them
        const material = new THREE.MeshLambertMaterial({
            color: 0xff9933,
            transparent: false,
            opacity: 1.0
        });
        
        const fcInputGroup = new THREE.Group();
        
        // Create spherical nodes and connection lines
        for (let i = 0; i < this.fcInputSize; i++) {
            // Create sphere
            const sphere = new THREE.Mesh(sphereGeometry, material.clone());
            
            // Position spheres in a vertical column, matching flattened layer spacing
            sphere.position.set(
                0, // Single column (x = 0)
                (i - (this.fcInputSize - 1) / 2) * this.spacing * 1.2, // Match flattened spacing
                0
            );
            
            // Store references
            sphere.fcInputIndex = i;
            sphere.connectedFlattenedIndex = i; // 1:1 mapping
            
            fcInputGroup.add(sphere);
            this.fcInputLayer.push(sphere);
            this.allFCInputNodes.push(sphere);
            this.originalFCInputMaterials.push(sphere.material.clone());
            
            // Create connection line from flattened cube to FC input sphere
            if (this.flattenedLayer[i]) {
                const flattenedCube = this.flattenedLayer[i];
                const flattenedPos = new THREE.Vector3();
                const spherePos = new THREE.Vector3();
                
                // Get world positions
                flattenedCube.getWorldPosition(flattenedPos);
                sphere.getWorldPosition(spherePos);
                
                // Adjust for group positions
                flattenedPos.x = 28; // Flattened layer X position
                spherePos.x = 36; // FC input layer X position
                
                const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                    new THREE.Vector3(28, flattenedPos.y, 0), // From flattened cube
                    new THREE.Vector3(36, spherePos.y, 0)     // To FC input sphere
                ]);
                
                const lineMaterial = new THREE.LineBasicMaterial({ 
                    color: 0x999999,
                    opacity: 0.6,
                    transparent: true
                });
                
                const line = new THREE.Line(lineGeometry, lineMaterial);
                this.connectionLines.push(line);
                this.scene.add(line);
            }
        }
        
        // Position the FC input layer
        const fcInputX = 36; // Distance from flattened layer
        fcInputGroup.position.set(fcInputX, 0, 0);
        
        this.scene.add(fcInputGroup);
    }
    
    createHiddenLayer() {
        this.clearHiddenLayer();
        
        const geometry = new THREE.SphereGeometry(0.3, 16, 16);
        
        // Green material for hidden layer nodes
        const material = new THREE.MeshLambertMaterial({
            color: 0x33cc33,
            transparent: false,
            opacity: 1.0
        });
        
        const hiddenGroup = new THREE.Group();
        
        // Create hidden layer nodes
        for (let i = 0; i < this.hiddenSize; i++) {
            const sphere = new THREE.Mesh(geometry, material.clone());
            
            // Add wireframe edges
            const edges = new THREE.EdgesGeometry(geometry);
            const edgeMaterial = new THREE.LineBasicMaterial({ 
                color: 0x000000,
                linewidth: 1 
            });
            const wireframe = new THREE.LineSegments(edges, edgeMaterial);
            sphere.add(wireframe);
            
            // Position nodes in a vertical column
            sphere.position.set(
                0,
                (i - (this.hiddenSize - 1) / 2) * 2.0, // Vertical spacing
                0
            );
            
            sphere.hiddenIndex = i;
            
            hiddenGroup.add(sphere);
            this.hiddenLayer.push(sphere);
            this.allHiddenNodes.push(sphere);
            this.originalHiddenMaterials.push(sphere.material.clone());
        }
        
        // Position the hidden layer
        const hiddenX = 38; // Distance from FC input layer
        hiddenGroup.position.set(hiddenX, 0, 0);
        
        this.scene.add(hiddenGroup);
    }
    
    createOutputNode() {
        this.clearOutputNode();
        
        const geometry = new THREE.SphereGeometry(0.4, 16, 16);
        
        // Red material for output node
        const material = new THREE.MeshLambertMaterial({
            color: 0xcc3333,
            transparent: false,
            opacity: 1.0
        });
        
        const outputGroup = new THREE.Group();
        
        // Create single output node
        const sphere = new THREE.Mesh(geometry, material.clone());
        
        // Add wireframe edges
        const edges = new THREE.EdgesGeometry(geometry);
        const edgeMaterial = new THREE.LineBasicMaterial({ 
            color: 0x000000,
            linewidth: 1 
        });
        const wireframe = new THREE.LineSegments(edges, edgeMaterial);
        sphere.add(wireframe);
        
        // Position at center
        sphere.position.set(0, 0, 0);
        sphere.outputIndex = 0;
        
        outputGroup.add(sphere);
        this.outputLayer.push(sphere);
        this.allOutputNodes.push(sphere);
        this.originalOutputMaterials.push(sphere.material.clone());
        
        // Position the output layer
        const outputX = 46; // Distance from hidden layer
        outputGroup.position.set(outputX, 0, 0);
        
        this.scene.add(outputGroup);
    }
    
    clearFCInputLayer() {
        // Remove existing FC input layer from scene
        this.fcInputLayer.forEach(sphere => {
            if (sphere.parent) {
                sphere.parent.remove(sphere);
            }
        });
        
        // Remove connection lines
        this.connectionLines.forEach(line => {
            this.scene.remove(line);
        });
        
        // Clear arrays
        this.fcInputLayer = [];
        this.allFCInputNodes = [];
        this.originalFCInputMaterials = [];
        this.connectionLines = [];
    }
    
    clearHiddenLayer() {
        // Remove existing hidden layer from scene
        this.hiddenLayer.forEach(node => {
            if (node.parent) {
                node.parent.remove(node);
            }
        });
        
        // Clear arrays
        this.hiddenLayer = [];
        this.allHiddenNodes = [];
        this.originalHiddenMaterials = [];
    }
    
    clearOutputNode() {
        // Remove existing output node from scene
        this.outputLayer.forEach(node => {
            if (node.parent) {
                node.parent.remove(node);
            }
        });
        
        // Clear arrays
        this.outputLayer = [];
        this.allOutputNodes = [];
        this.originalOutputMaterials = [];
    }
    
    updateExplanationText() {
        const kernelInfoElement = document.getElementById('kernelInfo');
        const kernelCountInfoElement = document.getElementById('kernelCountInfo');
        
        if (kernelInfoElement) {
            kernelInfoElement.textContent = 
                `Using ${this.numKernels} kernel${this.numKernels > 1 ? 's' : ''} creates ${this.numKernels} grayscale feature map${this.numKernels > 1 ? 's' : ''}, each highlighting different aspects of the input image.`;
        }
        
        if (kernelCountInfoElement) {
            const flattenedSize = this.pooledSize * this.pooledSize * this.numKernels;
            kernelCountInfoElement.textContent = 
                `Current setup: 8×8×3 input → ${this.numKernels} kernels → ${this.numKernels} feature maps (8×8) → max pool → ${this.numKernels} pooled maps (4×4) → flatten → ${flattenedSize} neurons → FC input (${flattenedSize} nodes)`;
        }
    }
    
    setupControls() {
        // Number of kernels slider
        const numKernelsSlider = document.getElementById('numKernels');
        const numKernelsValue = document.getElementById('numKernelsValue');
        
        if (numKernelsSlider) {
            numKernelsSlider.addEventListener('input', (e) => {
                this.numKernels = parseInt(e.target.value);
                numKernelsValue.textContent = e.target.value;
                this.createOutputLayers();
                this.createPooledLayers();
                this.createFlattenedLayer();
                this.createFCInputLayer();
                this.createHiddenLayer();
                this.createOutputNode();
                this.createConnections();
                this.updateExplanationText();
            });
        }
        
        // Reset view button
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.camera.position.set(38, 15, 20); // Reset to center-focused position
                this.controls.target.set(23, 0, 0);
                this.controls.reset();
            });
        }
        
        // Auto rotate button
        const rotateBtn = document.getElementById('rotateBtn');
        if (rotateBtn) {
            rotateBtn.addEventListener('click', () => {
                this.autoRotate = !this.autoRotate;
                rotateBtn.textContent = this.autoRotate ? 'Stop Rotation' : 'Auto Rotate';
                rotateBtn.classList.toggle('paused', this.autoRotate);
            });
        }
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Auto rotation
        if (this.autoRotate) {
            this.controls.autoRotate = true;
            this.controls.autoRotateSpeed = 2.0;
        } else {
            this.controls.autoRotate = false;
        }
        
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    createConnections() {
        this.clearConnections();
        
        const lineMaterial = new THREE.LineBasicMaterial({ 
            color: 0x666666, 
            transparent: true,
            opacity: 0.3
        });
        
        // Connect flattened layer to FC input layer
        this.flattenedLayer.forEach((flatCube, flatIndex) => {
            const fcIndex = flatIndex % this.fcInputSize; // Map to available FC nodes
            const fcNode = this.fcInputLayer[fcIndex];
            
            if (fcNode) {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    flatCube.getWorldPosition(new THREE.Vector3()),
                    fcNode.getWorldPosition(new THREE.Vector3())
                ]);
                
                const line = new THREE.Line(geometry, lineMaterial);
                this.connectionLines.push(line);
                this.scene.add(line);
            }
        });
        
        // Connect FC input layer to hidden layer
        this.fcInputLayer.forEach(fcNode => {
            this.hiddenLayer.forEach(hiddenNode => {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    fcNode.getWorldPosition(new THREE.Vector3()),
                    hiddenNode.getWorldPosition(new THREE.Vector3())
                ]);
                
                const line = new THREE.Line(geometry, lineMaterial);
                this.connectionLines.push(line);
                this.scene.add(line);
            });
        });
        
        // Connect hidden layer to output node
        this.hiddenLayer.forEach(hiddenNode => {
            this.outputLayer.forEach(outputNode => {
                const geometry = new THREE.BufferGeometry().setFromPoints([
                    hiddenNode.getWorldPosition(new THREE.Vector3()),
                    outputNode.getWorldPosition(new THREE.Vector3())
                ]);
                
                const line = new THREE.Line(geometry, lineMaterial);
                this.connectionLines.push(line);
                this.scene.add(line);
            });
        });
    }
    
    clearConnections() {
        // Remove existing connections from scene
        this.connectionLines.forEach(line => {
            if (line.parent) {
                line.parent.remove(line);
            } else {
                this.scene.remove(line);
            }
        });
        
        // Clear array
        this.connectionLines = [];
    }
}

// Initialize the visualization when the page loads
window.addEventListener('load', () => {
    new CNNVisualization();
});
