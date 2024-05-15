let displayActualTime = true;


function dragEnterHandler(event) {
    event.preventDefault();
}

function dragOverHandler(event) {
    event.preventDefault();
}


function switch_mode(event){
    console.log(event);
    const switch_button = document.getElementById("button_for_switch_mode");
    const dropArea = document.getElementById("dropArea")
    const analog_clock = document.getElementById("clock")
    const original_image = document.getElementById('original_image');
    const formed_image = document.getElementById('formed_image')
    const digital_clock = document.getElementById("digital_clock")
    const area_for_preprocessed_image = document.getElementById("preprocessed_image")
    const area_for_clear_image = document.getElementById("clear_image")
    const second_arrow = document.getElementById("sec_arrow")
    const clock_structure = document.getElementById("clock-struct")

    if (dropArea.ondrop === dropHandler_for_preprocessing){
        dropArea.ondrop = dropHandler_for_NN;
        switch_button.innerText = "Full Mode";
        analog_clock.className = "active"
        area_for_clear_image.className = "not_active"
        digital_clock.className = "active"
        area_for_preprocessed_image.className = "not_active"
        original_image.className = "not_active"
        formed_image.className = "not_active"
        second_arrow.className = 'not_active'
        clock_structure.className = 'clock_struct_for_NN'



    }
    else {
        dropArea.ondrop = dropHandler_for_preprocessing;
        switch_button.innerText = "NN Mode";
        analog_clock.className = "not_active"
        area_for_clear_image.className = "active"

        area_for_preprocessed_image.className = "active"
        original_image.className = "not_active"
        formed_image.className = 'not_active'
        second_arrow.className = 'active'
        clock_structure.className = 'clock_struct_for_full'
    }




}

function displayTime(time) {
    let digital_clock = document.querySelector('#digital_clock');
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

    const area_for_clear_image = document.getElementById("clear_image")
    const area_for_processed_image = document.getElementById("preprocessed_image")
    const original_image = document.getElementById('original_image');
    const formed_image = document.getElementById('formed_image')
    formed_image.className = "active"
    original_image.className = 'active'
    area_for_clear_image.className = "not_active"
    area_for_processed_image.className = "not_active"

    fetch('http://127.0.0.1:7777/upload_image_for_full', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            return response.json();
        })
        .then(data => {
            displayTime(data);
            display_result(getResult(data))
            display_result_mode()
            displayActualTime = false;
        })

        .catch(error => {
            console.error('Error:', error);
        });


    fetch('http://127.0.0.1:7777/output_image', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            console.log(blob)
            formed_image.src = URL.createObjectURL(blob);
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });

    fetch('http://127.0.0.1:7777/image', {
        method: 'POST',
        body: formData,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.blob();
        })
        .then(blob => {
            console.log(blob)
            original_image.src = URL.createObjectURL(blob);
        })
        .catch(error => {
            console.error('There has been a problem with your fetch operation:', error);
        });





}


function display_result(result){
    const hour_result = document.getElementById("hour_result")
    const minute_result = document.getElementById("minutes_result")
    const correct_result = document.getElementById("correct_result")
    const not_correct_result = document.getElementById("not_correct_result")

    if (result.hour === 0 && result.minute === 0){
        correct_result.className = "active"
        not_correct_result.className = "not_active"
        hour_result.className = "not_active"
        minute_result.className = "not_active"
    }
    else {

        correct_result.className = "not_active"
        not_correct_result.className = "active_for_button"
        hour_result.className = "active"
        minute_result.className = "active"
        if(result.hour > 0){
            hour_result.innerText = `You need to turn the hour hand forward by ${result.hour} hours.`
        }
        else {
            hour_result.innerText = `You need to turn the hour hand back by ${Math.abs(result.hour)} hours.`
        }

        if(result.minute > 0){
            minute_result.innerText = `You need to turn the minute hand forward by ${result.minute} minutes.`
        }
        else {
            minute_result.innerText = `You need to turn the minute hand back by ${Math.abs(result.minute)} minutes.`
        }
    }
}

function getResult(time){
    let date = new Date();
    let hour = date.getHours() - time.hour
    let minutes = date.getMinutes() - time.minute
    return {
        "hour": hour,
        "minute": minutes
    }
}

function back_button(event){
    const area_for_clear_image = document.getElementById("clear_image")
    const area_for_processed_image = document.getElementById("preprocessed_image")
    const result_block = document.getElementById("result_block")
    const original_image = document.getElementById('original_image');
    const formed_image = document.getElementById('formed_image')
    const dropArea = document.getElementById("dropArea")
    const button_back = document.getElementById("button_back")
    const button_for_switch_mode = document.getElementById("button_for_switch_mode")

    button_back.className = "not_active";
    original_image.className = "not_active"
    formed_image.className = "not_active"
    dropArea.className = "active"
    button_for_switch_mode.className = "active_for_button"
    result_block.className = "not_active"
    area_for_clear_image.className = "active"
    area_for_processed_image.className = "active"

}

function display_result_mode(){
    const result_block = document.getElementById("result_block")
    const original_image = document.getElementById('original_image');
    const formed_image = document.getElementById('formed_image')
    const dropArea = document.getElementById("dropArea")
    const button_back = document.getElementById("button_back")
    const button_for_switch_mode = document.getElementById("button_for_switch_mode")
    button_back.className = "active_for_button";
    original_image.className = "active"
    formed_image.className = "active"
    dropArea.className = "not_active"
    button_for_switch_mode.className = "not_active"
    result_block.className = "active"
}


setInterval(() => {
    if (!displayActualTime) return;
    let date = new Date();
    let hour = date.getHours() < 10 ? '0' + date.getHours() : date.getHours();
    let minute = date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes();

    displayTime({hour: hour, minute: minute})
}, 100)
