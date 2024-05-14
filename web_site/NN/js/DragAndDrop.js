let displayActualTime = true;


function dragEnterHandler(event) {
    event.preventDefault();
}

function dragOverHandler(event) {
    event.preventDefault();
}


function displayTime(time) {
    let digital_clock = document.querySelector('.digital_clock');
    const hoursArrow = document.querySelector('.hours');
    const minutesArrow = document.querySelector('.minutes');
    const deg = 6;

    // Calculate the rotation angles for the hour and minute arrows
    // The hour arrow rotates 30 degrees per hour plus an additional 0.5 degrees per minute
    // The minute arrow rotates 6 degrees per minute
    const hourRotation = (time.hour % 12) * 30 + (time.minute / 2);
    const minuteRotation = time.minute * deg;

    // Apply the rotation transformations to the clock arrows
    hoursArrow.style.transform = `rotateZ(${hourRotation}deg)`;
    minutesArrow.style.transform = `rotateZ(${minuteRotation}deg)`;

    // Update the digital clock display with the time
    digital_clock.innerHTML = `${String(time.hour).padStart(2, '0')}:${String(time.minute).padStart(2, '0')}`;
}


function dropHandler_for_NN(event) {
    console.log(event);
    event.preventDefault();

    const files = event.dataTransfer.files;
    const formData = new FormData();
    formData.append('image', files[0]);

    fetch('http://127.0.0.1:7777/upload_image_for_NN', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            return response.json();
        })
        .then(data => {
            displayTime(data);
            displayActualTime = false;
            // const time = `${data.hour}:${data.minute}`;
            //
            // // Update the content of the timeDisplay element
            // document.getElementById('timeDisplay').textContent = time;
        })

        .catch(error => {
            console.error('Error:', error);
        });
}


function dropHandler_for_preprocessing(event) {
    console.log(event);
    event.preventDefault();

    const files = event.dataTransfer.files;
    const formData = new FormData();
    formData.append('image', files[0]);

    fetch('http://127.0.0.1:7777/upload_image_for_preprocessing', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            return 0;
        })
}


setInterval(() => {
    if (!displayActualTime) return;
    let date = new Date();
    let hour = date.getHours() < 10 ? '0' + date.getHours() : date.getHours();
    let minute = date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes();

    displayTime({hour: hour, minute: minute})
}, 1000)
