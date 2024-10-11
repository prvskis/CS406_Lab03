document.addEventListener('DOMContentLoaded', function() {
    var fileInput = document.getElementById('file-upload');
    var fileNameDisplay = document.getElementById('file-name');
    
    if (fileInput && fileNameDisplay) {
        fileInput.addEventListener('change', function() {
            var fileName = this.files[0].name;
            fileNameDisplay.textContent = fileName;
        });
    }
});
