/**
 * MonuVision AI - Frontend Application
 * Three.js particle globe + API integration
 */

// ============================================
// Configuration
// ============================================
const API_URL = 'http://localhost:5000';
const PARTICLE_COUNT = 1500;

// ============================================
// Three.js Scene Setup
// ============================================
class ParticleGlobe {
    constructor(container) {
        this.container = container;
        this.mouse = { x: 0, y: 0 };
        this.targetRotation = { x: 0, y: 0 };

        this.init();
        this.createParticles();
        this.addEventListeners();
        this.animate();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();

        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.z = 5;

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.container.appendChild(this.renderer.domElement);
    }

    createParticles() {
        // Geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(PARTICLE_COUNT * 3);
        const colors = new Float32Array(PARTICLE_COUNT * 3);
        const sizes = new Float32Array(PARTICLE_COUNT);

        // Neon color palette
        const colorPalette = [
            new THREE.Color(0xff2d95), // Pink
            new THREE.Color(0x00f5ff), // Cyan
            new THREE.Color(0xb14aff), // Purple
            new THREE.Color(0x00ff88), // Green
        ];

        for (let i = 0; i < PARTICLE_COUNT; i++) {
            // Sphere distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const radius = 2 + Math.random() * 0.5;

            positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = radius * Math.cos(phi);

            // Random color from palette
            const color = colorPalette[Math.floor(Math.random() * colorPalette.length)];
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;

            // Random size
            sizes[i] = Math.random() * 3 + 1;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        // Material
        const material = new THREE.PointsMaterial({
            size: 0.03,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending,
            sizeAttenuation: true
        });

        // Points
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);

        // Add connecting lines for some particles
        this.createConnections();
    }

    createConnections() {
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = [];
        const lineColors = [];

        // Create random connections
        for (let i = 0; i < 100; i++) {
            const theta1 = Math.random() * Math.PI * 2;
            const phi1 = Math.acos(2 * Math.random() - 1);
            const theta2 = theta1 + (Math.random() - 0.5) * 0.5;
            const phi2 = phi1 + (Math.random() - 0.5) * 0.5;
            const radius = 2.2;

            linePositions.push(
                radius * Math.sin(phi1) * Math.cos(theta1),
                radius * Math.sin(phi1) * Math.sin(theta1),
                radius * Math.cos(phi1),
                radius * Math.sin(phi2) * Math.cos(theta2),
                radius * Math.sin(phi2) * Math.sin(theta2),
                radius * Math.cos(phi2)
            );

            // Purple to cyan gradient
            lineColors.push(0.69, 0.29, 1, 0, 0.96, 1);
        }

        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        lineGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

        const lineMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.2,
            blending: THREE.AdditiveBlending
        });

        this.lines = new THREE.LineSegments(lineGeometry, lineMaterial);
        this.scene.add(this.lines);
    }

    addEventListeners() {
        window.addEventListener('resize', () => this.onResize());
        window.addEventListener('mousemove', (e) => this.onMouseMove(e));
    }

    onResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    onMouseMove(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Smooth rotation following mouse
        this.targetRotation.x = this.mouse.y * 0.3;
        this.targetRotation.y = this.mouse.x * 0.3;

        this.particles.rotation.x += (this.targetRotation.x - this.particles.rotation.x) * 0.05;
        this.particles.rotation.y += (this.targetRotation.y - this.particles.rotation.y) * 0.05;

        // Auto rotation
        this.particles.rotation.y += 0.001;

        // Sync lines rotation
        if (this.lines) {
            this.lines.rotation.copy(this.particles.rotation);
        }

        this.renderer.render(this.scene, this.camera);
    }
}

// ============================================
// UI Controller
// ============================================
class MonuVisionApp {
    constructor() {
        this.uploadZone = document.getElementById('upload-zone');
        this.fileInput = document.getElementById('file-input');
        this.previewContainer = document.getElementById('preview-container');
        this.previewImage = document.getElementById('preview-image');
        this.loadingContainer = document.getElementById('loading-container');
        this.resultsContainer = document.getElementById('results-container');
        this.resultTitle = document.getElementById('result-title');
        this.confidenceFill = document.getElementById('confidence-fill');
        this.confidenceValue = document.getElementById('confidence-value');
        this.predictionsList = document.getElementById('predictions-list');
        this.clearBtn = document.getElementById('clear-btn');
        this.tryAgainBtn = document.getElementById('try-again-btn');

        this.currentFile = null;

        this.initEventListeners();
    }

    initEventListeners() {
        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Upload zone click
        this.uploadZone.addEventListener('click', () => this.fileInput.click());

        // Drag and drop
        this.uploadZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.uploadZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.uploadZone.addEventListener('drop', (e) => this.handleDrop(e));

        // Clear button
        this.clearBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.reset();
        });

        // Try again button
        this.tryAgainBtn.addEventListener('click', () => this.reset());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadZone.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadZone.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith('image/')) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    processFile(file) {
        this.currentFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImage.src = e.target.result;
            this.showPreview();
            this.uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    showPreview() {
        this.uploadZone.style.display = 'none';
        this.previewContainer.style.display = 'block';
        this.resultsContainer.style.display = 'none';
    }

    showLoading() {
        this.previewContainer.style.display = 'none';
        this.loadingContainer.style.display = 'flex';
    }

    showResults(data) {
        this.loadingContainer.style.display = 'none';
        this.resultsContainer.style.display = 'block';
        this.previewContainer.style.display = 'block';

        // Check if monument was recognized (confidence >= 80%)
        const isRecognized = data.recognized !== false;

        if (isRecognized) {
            // Monument recognized - show normal result
            this.resultTitle.textContent = data.top_class || data.predictions[0].class;
            this.resultTitle.style.background = 'var(--gradient-primary)';
            this.resultTitle.style.webkitBackgroundClip = 'text';
            this.resultTitle.style.webkitTextFillColor = 'transparent';

            // Animate confidence bar
            const confidence = data.top_confidence || data.predictions[0].confidence;
            setTimeout(() => {
                this.confidenceFill.style.width = `${confidence * 100}%`;
                this.confidenceFill.style.background = 'var(--gradient-primary)';
            }, 100);
            this.confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
            this.confidenceValue.style.color = 'var(--neon-green)';
        } else {
            // Monument NOT recognized - show "Not Recognized" state
            this.resultTitle.textContent = '❌ Not Recognized';
            this.resultTitle.style.background = 'linear-gradient(135deg, #ff4757, #ff6b81)';
            this.resultTitle.style.webkitBackgroundClip = 'text';
            this.resultTitle.style.webkitTextFillColor = 'transparent';

            // Show low confidence bar in red
            const confidence = data.top_confidence;
            setTimeout(() => {
                this.confidenceFill.style.width = `${confidence * 100}%`;
                this.confidenceFill.style.background = 'linear-gradient(135deg, #ff4757, #ff6b81)';
            }, 100);
            this.confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
            this.confidenceValue.style.color = '#ff4757';
        }

        // Update predictions list
        this.predictionsList.innerHTML = '';

        // Add message if not recognized
        if (!isRecognized && data.message) {
            const messageItem = document.createElement('p');
            messageItem.style.color = 'var(--neon-orange)';
            messageItem.style.marginBottom = '1rem';
            messageItem.textContent = data.message;
            this.predictionsList.appendChild(messageItem);
        }

        (data.predictions || []).slice(0, 5).forEach((pred, index) => {
            const item = document.createElement('div');
            item.className = 'prediction-item';
            item.innerHTML = `
                <span class="prediction-rank">${index + 1}</span>
                <span class="prediction-name">${pred.class}</span>
                <span class="prediction-confidence">${(pred.confidence * 100).toFixed(1)}%</span>
            `;
            this.predictionsList.appendChild(item);
        });
    }

    showError(message) {
        this.loadingContainer.style.display = 'none';
        this.previewContainer.style.display = 'block';

        // Show error in results
        this.resultsContainer.style.display = 'block';
        this.resultTitle.textContent = 'Error';
        this.confidenceFill.style.width = '0%';
        this.confidenceValue.textContent = '—';
        this.predictionsList.innerHTML = `<p style="color: var(--neon-pink);">${message}</p>`;
    }

    async uploadImage(file) {
        this.showLoading();

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const data = await response.json();
            this.showResults(data);
        } catch (error) {
            console.error('Error:', error);
            this.showError('Could not connect to the AI server. Make sure the API is running.');
        }
    }

    reset() {
        this.currentFile = null;
        this.fileInput.value = '';
        this.uploadZone.style.display = 'block';
        this.previewContainer.style.display = 'none';
        this.loadingContainer.style.display = 'none';
        this.resultsContainer.style.display = 'none';
        this.confidenceFill.style.width = '0%';
    }
}

// ============================================
// Initialize Application
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js particle globe
    const threeContainer = document.getElementById('three-container');
    new ParticleGlobe(threeContainer);

    // Initialize app controller
    new MonuVisionApp();

    console.log('🏛️ MonuVision AI initialized!');
});
