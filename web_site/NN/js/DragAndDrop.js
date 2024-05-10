function dragEnterHandler(event) {
    event.preventDefault();
}

function dragOverHandler(event) {
    event.preventDefault();
}

function dropHandler(event) {
    console.log(event);
    event.preventDefault();

    const files = event.dataTransfer.files;
    const formData = new FormData();
    formData.append('image', files[0]);

    fetch('http://127.0.0.1:8000/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            // Handle the response here
        })
        .catch(error => {
            console.error('Error:', error);
        });
}
