async function predict() {

    const imageInput = document.getElementById("imageInput");
    const modelSelect = document.getElementById("modelSelect");

    if (imageInput.files.length === 0) {
        alert("Please select an image");
        return;
    }

    const file = imageInput.files[0];
    const model = modelSelect.value;

    const formData = new FormData();
    formData.append("image", file);
    formData.append("model", model);

    document.getElementById("result").innerHTML = "Processing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

if (!response.ok) {

    document.getElementById("result").innerHTML =
        `<h3>${data.error}</h3>`;

    return;
}

        document.getElementById("result").innerHTML = `
            <h3>Prediction: ${data.prediction}</h3>
            <h4>Confidence: ${(data.confidence * 100).toFixed(2)}%</h4>
        `;

        document.getElementById("gradcamImage").src =
            "data:image/jpeg;base64," + data.gradcam;

    } catch (error) {
        console.error(error);
        document.getElementById("result").innerHTML = "Error occurred";
    }
}

