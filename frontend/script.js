document.addEventListener('DOMContentLoaded', () => {
    const uploadInput = document.getElementById('image-upload');
    const predictBtn = document.getElementById('predict-btn');
    const previewImg = document.getElementById('preview');
    const digitResult = document.getElementById('digit-result');
    const confidenceChart = document.getElementById('confidence-chart');
    
    // Handle image preview
    uploadInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                previewImg.src = event.target.result;
                previewImg.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Handle prediction
   predictBtn.addEventListener('click', async () => {
    const file = uploadInput.files[0];
    if (!file) {
        alert('Please select an image first!');
        return;
    }
    
    try {
        digitResult.textContent = '...';
        confidenceChart.innerHTML = '<p>Loading...</p>';
        
        const formData = new FormData();
        formData.append('image', file);
        
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        // First check if response is OK
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        digitResult.textContent = result.digit;
        renderConfidenceChart(result.probabilities);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error: ${error.message}`);
    }
});
    
    // Render confidence chart
    function renderConfidenceChart(probabilities) {
        confidenceChart.innerHTML = '';
        
        probabilities.forEach((percent, digit) => {
            const barContainer = document.createElement('div');
            barContainer.className = 'confidence-bar';
            
            const barFill = document.createElement('div');
            barFill.className = 'bar-fill';
            barFill.style.width = `${percent}%`;
            
            const digitLabel = document.createElement('div');
            digitLabel.className = 'digit-label';
            digitLabel.textContent = digit;
            
            const percentLabel = document.createElement('div');
            percentLabel.textContent = `${percent.toFixed(1)}%`;
            
            barContainer.appendChild(barFill);
            barContainer.appendChild(percentLabel);
            barContainer.appendChild(digitLabel);
            
            confidenceChart.appendChild(barContainer);
        });
    }
});