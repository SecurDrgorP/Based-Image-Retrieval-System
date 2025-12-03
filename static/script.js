let selectedImage = {
    shapes: null,
    textures: null
};

function switchTab(tab) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    event.target.classList.add('active');
    document.getElementById(`${tab}-tab`).classList.add('active');
    
    if (!selectedImage[tab]) {
        loadImages(tab);
    }
}

async function loadImages(type) {
    const grid = document.getElementById(`${type}-grid`);
    grid.innerHTML = '<div class="loading">Loading images...</div>';
    
    try {
        const response = await fetch(`/api/images/${type}`);
        const images = await response.json();
        
        grid.innerHTML = '';
        images.forEach(img => {
            const card = document.createElement('div');
            card.className = 'image-card';
            card.onclick = () => selectImage(type, img);
            card.innerHTML = `
                <img src="/images/${type === 'shapes' ? 'Formes' : 'Textures'}/${img}" alt="${img}">
                <p>${img}</p>
            `;
            grid.appendChild(card);
        });
    } catch (error) {
        grid.innerHTML = `<div class="error">Error loading images: ${error.message}</div>`;
    }
}

function selectImage(type, image) {
    selectedImage[type] = image;
    
    document.querySelectorAll(`#${type}-grid .image-card`).forEach(card => {
        card.classList.remove('selected');
    });
    
    event.currentTarget.classList.add('selected');
}

async function extractFeatures(type) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Extracting...';
    
    try {
        const response = await fetch(`/api/extract/${type}`, {
            method: 'POST'
        });
        const data = await response.json();
        
        if (data.success) {
            alert(data.message);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Extract All Features';
    }
}

async function search(type) {
    const resultsDiv = document.getElementById(`${type}-results`);
    
    if (!selectedImage[type]) {
        alert('Please select an image first');
        return;
    }
    
    resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
    
    try {
        const response = await fetch(`/api/search/${type}`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image: selectedImage[type],
                top_k: 6
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(type, data.query, data.results);
        } else {
            resultsDiv.innerHTML = `<div class="error">${data.error}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

function displayResults(type, query, results) {
    const resultsDiv = document.getElementById(`${type}-results`);
    
    let html = `<h2>Results for: ${query}</h2>`;
    
    results.forEach((result, index) => {
        html += `
            <div class="result-card">
                <img src="${result.path}" alt="${result.name}">
                <div class="result-info">
                    <h3>${index + 1}. ${result.name}</h3>
                    <p class="distance">Distance: ${result.distance.toFixed(6)}</p>
                    <div class="similarity-bar">
                        <div class="similarity-fill" style="width: ${result.similarity}%">
                            ${result.similarity.toFixed(1)}%
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

// Load shape images on page load
loadImages('shapes');