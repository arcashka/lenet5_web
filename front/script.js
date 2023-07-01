let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let captureButton = document.getElementById('capture');
let notification = document.getElementById('notification');
let context = canvas.getContext('2d');
let serverUrl = "https://192.168.1.8:5000/process";  // Modify this according to your setup

let currentStream;

function stopMediaTracks(stream) {
    stream.getTracks().forEach(track => {
        track.stop();
    });
}

// Get camera devices and set the initial camera
navigator.mediaDevices.enumerateDevices().then(devices => {
    let cameras = devices.filter(device => device.kind === 'videoinput');
    if (cameras && cameras.length > 0) {
        let selectedCamera = cameras[0];

        // Set the initial camera
        const constraints = {
            video: {
                deviceId: selectedCamera.deviceId
            }
        };

        navigator.mediaDevices.getUserMedia(constraints)
            .then((stream) => {
                video.srcObject = stream;
                currentStream = stream;
            });

        const switchButton = document.getElementById('switch');

        // Event listener for the switch button
        switchButton.addEventListener('click', () => {
            if (currentStream) {
                stopMediaTracks(currentStream);
            }

            if (cameras.length > 1) {
                // Switch to the next camera
                selectedCamera = (selectedCamera === cameras[0]) ? cameras[2] : cameras[0];
                const constraints = {
                    video: {
                        deviceId: selectedCamera.deviceId
                    }
                };

                navigator.mediaDevices.getUserMedia(constraints)
                    .then((stream) => {
                        video.srcObject = stream;
                        currentStream = stream;
                    });
            }
        });
    }
});

captureButton.addEventListener('click', () => {
    if (canvas.style.display === 'none') {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.style.display = 'block';
        video.style.display = 'none';
        sendPhotoToServer();
    } else {
        canvas.style.display = 'none';
        video.style.display = 'block';
        notification.style.display = 'none';
    }
});

function sendPhotoToServer() {
    canvas.toBlob(function(blob) {
        let formData = new FormData();
        formData.append('image', blob);
        let options = {
            method: 'POST',
            body: formData,
        };
        fetch(serverUrl, options)
            .then(response => response.json())
            .then(data => {
                notification.textContent = data.result;
                notification.style.display = 'block';
                let processedImg = new Image();
                processedImg.onload = function() {
                    context.clearRect(0, 0, canvas.width, canvas.height);
                    context.drawImage(processedImg, 0, 0, canvas.width, canvas.height);
                };
                processedImg.src = 'data:image/jpeg;base64,' + data.processed_image;
            })
            .catch((error) => {
                console.error('Error:', error);
            });
    }, 'image/png');
}
