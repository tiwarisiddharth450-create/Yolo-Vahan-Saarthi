document.addEventListener('DOMContentLoaded', () => {
    const uploadDropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    const mediaContainer = document.getElementById('mediaContainer');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');

    // Trigger file dialog
    uploadBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadDropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // File input changes
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUpload(e.target.files[0]);
        }
    });

    // Drag and Drop styling
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadDropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadDropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadDropzone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUpload(files[0]);
        }
    });

    function handleUpload(file) {
        // Validate file types
        const isImage = file.type.startsWith('image/');
        const isVideo = file.type.startsWith('video/');

        if (!isImage && !isVideo) {
            showError('Invalid file format. Please upload an image or video.');
            return;
        }

        // Setup FormData
        const formData = new FormData();
        formData.append('file', file);

        // UI States: show loading, hide upload dropzone and previous results
        uploadDropzone.classList.add('hidden');
        resultSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || 'Server error during analysis'); });
                }
                return response.json();
            })
            .then(data => {
                loadingSection.classList.add('hidden');
                renderResult(data);
            })
            .catch(err => {
                loadingSection.classList.add('hidden');
                uploadDropzone.classList.remove('hidden');
                showError(err.message);
            });
    }

    function renderResult(data) {
        mediaContainer.innerHTML = '';

        if (data.type === 'video') {
            const video = document.createElement('video');
            video.className = 'result-media';
            video.controls = true;
            video.autoplay = true;
            video.loop = true;
            video.muted = true;
            video.playsInline = true;
            // Add source element with explicit MIME type for Chrome compatibility
            const source = document.createElement('source');
            source.src = data.url + '?t=' + Date.now(); // cache-bust to force fresh fetch
            source.type = 'video/mp4';
            video.appendChild(source);
            mediaContainer.appendChild(video);
            video.load(); // Required when src is set dynamically
        } else {
            const img = document.createElement('img');
            img.src = data.url;
            img.className = 'result-media';
            img.alt = 'Analyzed result';
            mediaContainer.appendChild(img);
        }

        // Enable download button
        downloadCsvBtn.href = data.csv_url;

        // Reveal results and let dropzone remain hidden or reveal for another run
        resultSection.classList.remove('hidden');

        // Show dropzone below results for easy upload of a new file
        uploadDropzone.classList.remove('hidden');
    }

    function showError(message) {
        // Simple and elegant error presentation
        const errorDiv = document.createElement('div');
        errorDiv.style.position = 'fixed';
        errorDiv.style.bottom = '20px';
        errorDiv.style.right = '20px';
        errorDiv.style.background = 'rgba(239, 68, 68, 0.9)';
        errorDiv.style.color = '#ffffff';
        errorDiv.style.padding = '1rem 1.5rem';
        errorDiv.style.borderRadius = '12px';
        errorDiv.style.boxShadow = '0 10px 25px rgba(239, 68, 68, 0.3)';
        errorDiv.style.backdropFilter = 'blur(10px)';
        errorDiv.style.zIndex = '1000';
        errorDiv.style.transition = 'opacity 0.5s ease';
        errorDiv.textContent = message;

        document.body.appendChild(errorDiv);

        setTimeout(() => {
            errorDiv.style.opacity = '0';
            setTimeout(() => errorDiv.remove(), 500);
        }, 5000);
    }
});
