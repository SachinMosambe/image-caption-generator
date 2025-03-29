document.getElementById("imageForm").addEventListener("submit", function(event) {
    event.preventDefault();  // Prevent form submission

    var formData = new FormData();
    var fileInput = document.getElementById("imageInput");

    // Ensure file is selected
    if (fileInput.files.length === 0) {
        alert("Please select an image.");
        return;
    }

    formData.append("image", fileInput.files[0]);

    // Show loading spinner
    document.getElementById("loading").style.display = "block";
    document.getElementById("captionOutput").style.display = "none";

    // Send POST request to Flask server
    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("loading").style.display = "none";
        document.getElementById("captionText").textContent = data.caption;
        document.getElementById("captionOutput").style.display = "block";
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById("loading").style.display = "none";
        alert("An error occurred. Please try again.");
    });
});
