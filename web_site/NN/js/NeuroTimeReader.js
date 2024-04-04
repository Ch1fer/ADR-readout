function clock() {
    const hoursArrow = document.querySelector('.hours')
    const minutesArrow = document.querySelector('.minutes')
    const secondsArrow = document.querySelector('.seconds')
    const deg = 6

    setInterval(() => {
        const day = new Date()
        const hours = day.getHours() * 30
        const minutes = day.getMinutes() * deg
        const seconds = day.getSeconds() * deg

        hoursArrow.style.transform = `rotateZ(${hours + (minutes / 12)}deg)`
        minutesArrow.style.transform = `rotateZ(${minutes}deg)`
        secondsArrow.style.transform = `rotateZ(${seconds}deg)`
    }, 1000)
}



let digital_clock = document.querySelector('.digital_clock');

function time() {
    let date = new Date();
    let hours = date.getHours() < 10 ? '0' + date.getHours() : date.getHours();
    let min = date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes();
    let sec = date.getSeconds() < 10 ? '0' + date.getSeconds() : date.getSeconds();

    digital_clock.innerHTML = `${hours}:${min}:${sec}`;
}
setInterval(time, 1000);

time();
clock()