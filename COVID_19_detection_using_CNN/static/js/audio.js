URL = window.URL || window.webkitURL;

var gumStream;                      //stream from getUserMedia()
var rec;                            //Recorder.js object
var input;                          //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record




var audio1 = document.getElementById("audio1")
var startrecordbtn1 = document.getElementById("startrecord1");
var playpaussample1 = document.getElementById("playpausebtn1")
var recordButton1 = document.getElementById("recordButton1");
var stopButton1 = document.getElementById("stopButton1");
var pauseButton1 = document.getElementById("pauseButton1");
var uploadbutton1 = document.getElementById('uploadButton1');
var one = document.getElementById('1')

var result = document.getElementById('finish')
// result.disabled = true

//add events to those 2 buttons
recordButton1.addEventListener("click", startRecording1);
stopButton1.addEventListener("click", stopRecording1);
pauseButton1.addEventListener("click", pauseRecording1);


result.addEventListener("click", getresult)
var count = 0
function myFunction1() {
    if (recordButton1.style.display == "none" && stopButton1.style.display == "none" && pauseButton1.style.display == "none") {
        console.log("display three button")
        startrecordbtn1.style.display = "none"
        playpaussample1.style.display = "none"
        recordButton1.style.display = "block";
        stopButton1.style.display = "block";
        pauseButton1.style.display = "block";
    } else {
        recordButton1.style.display = "none";
        stopButton1.style.display = "none";
        pauseButton1.style.display = "none";
    }
}

function playpause1() {
    if (count == 0) {
        count = 1
        audio1.play();
        playpaussample1.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio1.pause();
        playpaussample1.innerHTML = "Play &#9658";
    }
}

function startRecording1() {
    console.log("recordButton1 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton1.disabled = true;
    stopButton1.disabled = false;
    pauseButton1.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton1.disabled = false;
        stopButton1.disabled = true;
        pauseButton1.disabled = true
    });
}

function pauseRecording1() {
    console.log("pauseButton1 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton1.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton1.innerHTML = "Pause";

    }
}

function stopRecording1() {
    console.log("stopButton1 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton1.disabled = true;
    recordButton1.disabled = false;
    pauseButton1.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton1.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton1.style.display = "none"
    pauseButton1.style.display = "none"
    stopButton1.style.display = "none"

    uploadbutton1.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink1);
}

function createDownloadLink1(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = 'Breathing-shallow' 

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    // link.href = url;
    // link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
    // link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    // li.appendChild(document.createTextNode(filename+".wav "))

    //add the save to disk link to li
    // li.appendChild(link);

    //upload link

    uploadbutton1.href = "#";
    uploadbutton1.innerHTML = "Upload";
    uploadbutton1.addEventListener("click", function (event) {
        one.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        
        
    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton1)//add the upload link to li

    //add the li element to the ol
    recordingsList1.appendChild(li);
    // recordingsList1.style.display = 'none'
    // uploadbutton1.style.display = 'none'   
}

// ################################################################################################
// ################################################################################################
// ################################################################################################


var audio2 = document.getElementById("audio2")
var startrecordbtn2 = document.getElementById("startrecord2");
var playpaussample2 = document.getElementById("playpausebtn2")

var recordButton2 = document.getElementById("recordButton2");
var stopButton2 = document.getElementById("stopButton2");
var pauseButton2 = document.getElementById("pauseButton2");

var uploadbutton2 = document.getElementById('uploadButton2');
var two = document.getElementById('2')

//add events to those 2 buttons
recordButton2.addEventListener("click", startRecording2);
stopButton2.addEventListener("click", stopRecording2);
pauseButton2.addEventListener("click", pauseRecording2);

var count = 0
function myFunction2() {
    if (recordButton2.style.display == "none" && stopButton2.style.display == "none" && pauseButton2.style.display == "none") {
        console.log("display three button")
        startrecordbtn2.style.display = "none"
        playpaussample2.style.display = "none"
        recordButton2.style.display = "block";
        stopButton2.style.display = "block";
        pauseButton2.style.display = "block";
    } else {
        recordButton2.style.display = "none";
        stopButton2.style.display = "none";
        pauseButton2.style.display = "none";
    }
}

function playpause2() {
    if (count == 0) {
        count = 1
        audio2.play();
        playpaussample2.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample2.innerHTML = "Play &#9658";
    }
}

function startRecording2() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton2.disabled = true;
    stopButton2.disabled = false;
    pauseButton2.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton2.disabled = false;
        stopButton2.disabled = true;
        pauseButton2.disabled = true
    });
}

function pauseRecording2() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton2.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton2.innerHTML = "Pause";

    }
}

function stopRecording2() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton2.disabled = true;
    recordButton2.disabled = false;
    pauseButton2.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton2.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton2.style.display = "none"
    pauseButton2.style.display = "none"
    stopButton2.style.display = "none"

    uploadbutton2.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink2);
}

function createDownloadLink2(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = 'Breathing-deep' 
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    // link.href = url;
    // link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
    // link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    // li.appendChild(document.createTextNode(filename+".wav "))

    //add the save to disk link to li
    // li.appendChild(link);

    //upload link

    uploadbutton2.href = "#";
    uploadbutton2.innerHTML = "Upload";
    uploadbutton2.addEventListener("click", function (event) {
        two.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        
    
    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton2)//add the upload link to li

    //add the li element to the ol
    recordingsList2.appendChild(li);
    // recordingsList1.style.display = 'none'
    // uploadbutton2.style.display = 'none'
    
}

// ####################################################################################
// ####################################################################################
// ####################################################################################

var audio3 = document.getElementById("audio3")
var startrecordbtn3 = document.getElementById("startrecord3");
var playpaussample3 = document.getElementById("playpausebtn3")

var recordButton3 = document.getElementById("recordButton3");
var stopButton3 = document.getElementById("stopButton3");
var pauseButton3 = document.getElementById("pauseButton3");

var uploadbutton3 = document.getElementById('uploadButton3');
var three = document.getElementById('3')

//add events to those 2 buttons
recordButton3.addEventListener("click", startRecording3);
stopButton3.addEventListener("click", stopRecording3);
pauseButton3.addEventListener("click", pauseRecording3);

var count = 0
function myFunction3() {
    if (recordButton3.style.display == "none" && stopButton3.style.display == "none" && pauseButton3.style.display == "none") {
        console.log("display three button")
        startrecordbtn3.style.display = "none"
        playpaussample3.style.display = "none"
        recordButton3.style.display = "block";
        stopButton3.style.display = "block";
        pauseButton3.style.display = "block";
    } else {
        recordButton3.style.display = "none";
        stopButton3.style.display = "none";
        pauseButton3.style.display = "none";
    }
}

function playpause3() {
    if (count == 0) {
        count = 1
        audio3.play();
        playpaussample3.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample3.innerHTML = "Play &#9658";
    }
}

function startRecording3() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton3.disabled = true;
    stopButton3.disabled = false;
    pauseButton3.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton3.disabled = false;
        stopButton3.disabled = true;
        pauseButton3.disabled = true
    });
}

function pauseRecording3() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton3.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton3.innerHTML = "Pause";

    }
}

function stopRecording3() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton3.disabled = true;
    recordButton3.disabled = false;
    pauseButton3.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton3.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton3.style.display = "none"
    pauseButton3.style.display = "none"
    stopButton3.style.display = "none"

    uploadbutton3.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink3);
}

function createDownloadLink3(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Cough-heavy"
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    //save to disk link
    // link.href = url;
    // link.download = filename+".wav"; //download forces the browser to donwload the file using the  filename
    // link.innerHTML = "Save to disk";

    //add the new audio element to li
    li.appendChild(au);

    //add the filename to the li
    // li.appendChild(document.createTextNode(filename+".wav "))

    //add the save to disk link to li
    // li.appendChild(link);

    //upload link

    uploadbutton3.href = "#";
    uploadbutton3.innerHTML = "Upload";
    uploadbutton3.addEventListener("click", function (event) {
        three.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        
    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton3)//add the upload link to li

    //add the li element to the ol
    recordingsList3.appendChild(li);
    // recordingsList1.style.display = 'none'
    // uploadbutton2.style.display = 'none'

}

// ###############################################################################
// ###############################################################################
// ###############################################################################


var audio4 = document.getElementById("audio4")
var startrecordbtn4 = document.getElementById("startrecord4");
var playpaussample4 = document.getElementById("playpausebtn4")

var recordButton4 = document.getElementById("recordButton4");
var stopButton4 = document.getElementById("stopButton4");
var pauseButton4 = document.getElementById("pauseButton4");

var uploadbutton4 = document.getElementById('uploadButton4');
var four = document.getElementById('4')

//add events to those 2 buttons
recordButton4.addEventListener("click", startRecording4);
stopButton4.addEventListener("click", stopRecording4);
pauseButton4.addEventListener("click", pauseRecording4);

var count = 0
function myFunction4() {
    if (recordButton4.style.display == "none" && stopButton4.style.display == "none" && pauseButton4.style.display == "none") {
        console.log("display three button")
        startrecordbtn4.style.display = "none"
        playpaussample4.style.display = "none"
        recordButton4.style.display = "block";
        stopButton4.style.display = "block";
        pauseButton4.style.display = "block";
    } else {
        recordButton4.style.display = "none";
        stopButton4.style.display = "none";
        pauseButton4.style.display = "none";
    }
}

function playpause4() {
    if (count == 0) {
        count = 1
        audio4.play();
        playpaussample4.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample4.innerHTML = "Play &#9658";
    }
}

function startRecording4() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton4.disabled = true;
    stopButton4.disabled = false;
    pauseButton4.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton4.disabled = false;
        stopButton4.disabled = true;
        pauseButton4.disabled = true
    });
}

function pauseRecording4() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton4.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton4.innerHTML = "Pause";

    }
}

function stopRecording4() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton4.disabled = true;
    recordButton4.disabled = false;
    pauseButton4.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton4.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton4.style.display = "none"
    pauseButton4.style.display = "none"
    stopButton4.style.display = "none"

    uploadbutton4.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink4);
}

function createDownloadLink4(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Cough-shallow"
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton4.href = "#";
    uploadbutton4.innerHTML = "Upload";
    uploadbutton4.addEventListener("click", function (event) {
        four.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        

    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton4)//add the upload link to li

    //add the li element to the ol
    recordingsList4.appendChild(li);   
}


// #######################################################################################
// #######################################################################################

var audio5 = document.getElementById("audio5")
var startrecordbtn5 = document.getElementById("startrecord5");
var playpaussample5 = document.getElementById("playpausebtn5")

var recordButton5 = document.getElementById("recordButton5");
var stopButton5 = document.getElementById("stopButton5");
var pauseButton5 = document.getElementById("pauseButton5");

var uploadbutton5 = document.getElementById('uploadButton5');
var five = document.getElementById('5')

//add events to those 2 buttons
recordButton5.addEventListener("click", startRecording5);
stopButton5.addEventListener("click", stopRecording5);
pauseButton5.addEventListener("click", pauseRecording5);

var count = 0
function myFunction5() {
    if (recordButton5.style.display == "none" && stopButton5.style.display == "none" && pauseButton5.style.display == "none") {
        console.log("display three button")
        startrecordbtn5.style.display = "none"
        playpaussample5.style.display = "none"
        recordButton5.style.display = "block";
        stopButton5.style.display = "block";
        pauseButton5.style.display = "block";
    } else {
        recordButton5.style.display = "none";
        stopButton5.style.display = "none";
        pauseButton5.style.display = "none";
    }
}

function playpause5() {
    if (count == 0) {
        count = 1
        audio5.play();
        playpaussample5.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample5.innerHTML = "Play &#9658";
    }
}

function startRecording5() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton5.disabled = true;
    stopButton5.disabled = false;
    pauseButton5.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton5.disabled = false;
        stopButton5.disabled = true;
        pauseButton5.disabled = true
    });
}

function pauseRecording5() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton5.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton5.innerHTML = "Pause";

    }
}

function stopRecording5() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton5.disabled = true;
    recordButton5.disabled = false;
    pauseButton5.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton5.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton5.style.display = "none"
    pauseButton5.style.display = "none"
    stopButton5.style.display = "none"

    uploadbutton5.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink5);
}

function createDownloadLink5(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Counting-Normal"
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton5.href = "#";
    uploadbutton5.innerHTML = "Upload";
    uploadbutton5.addEventListener("click", function (event) {

        five.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        

    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton5)//add the upload link to li

    //add the li element to the ol
    recordingsList5.appendChild(li);
    
}

// ################################################################################################
// ################################################################################################
// ################################################################################################

var audio6 = document.getElementById("audio6")
var startrecordbtn6 = document.getElementById("startrecord6");
var playpaussample6 = document.getElementById("playpausebtn6")

var recordButton6 = document.getElementById("recordButton6");
var stopButton6 = document.getElementById("stopButton6");
var pauseButton6 = document.getElementById("pauseButton6");

var uploadbutton6 = document.getElementById('uploadButton6');
var six = document.getElementById('6')

//add events to those 2 buttons
recordButton6.addEventListener("click", startRecording6);
stopButton6.addEventListener("click", stopRecording6);
pauseButton6.addEventListener("click", pauseRecording6);

var count = 0
function myFunction6() {
    if (recordButton6.style.display == "none" && stopButton6.style.display == "none" && pauseButton6.style.display == "none") {
        console.log("display three button")
        startrecordbtn6.style.display = "none"
        playpaussample6.style.display = "none"
        recordButton6.style.display = "block";
        stopButton6.style.display = "block";
        pauseButton6.style.display = "block";
    } else {
        recordButton6.style.display = "none";
        stopButton6.style.display = "none";
        pauseButton6.style.display = "none";
    }
}

function playpause6() {
    if (count == 0) {
        count = 1
        audio6.play();
        playpaussample6.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample6.innerHTML = "Play &#9658";
    }
}

function startRecording6() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton6.disabled = true;
    stopButton6.disabled = false;
    pauseButton6.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton6.disabled = false;
        stopButton6.disabled = true;
        pauseButton6.disabled = true
    });
}

function pauseRecording6() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton6.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton6.innerHTML = "Pause";

    }
}

function stopRecording6() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton6.disabled = true;
    recordButton6.disabled = false;
    pauseButton6.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton6.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton6.style.display = "none"
    pauseButton6.style.display = "none"
    stopButton6.style.display = "none"

    uploadbutton6.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink6);
}

function createDownloadLink6(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Counting-fast"
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton6.href = "#";
    uploadbutton6.innerHTML = "Upload";
    uploadbutton6.addEventListener("click", function (event) {
        six.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton6)//add the upload link to li

    //add the li element to the ol
    recordingsList6.appendChild(li);
    

}

// #########################################################################################
// #########################################################################################
// #########################################################################################


var audio7 = document.getElementById("audio7")
var startrecordbtn7 = document.getElementById("startrecord7");
var playpaussample7 = document.getElementById("playpausebtn7")

var recordButton7 = document.getElementById("recordButton7");
var stopButton7 = document.getElementById("stopButton7");
var pauseButton7 = document.getElementById("pauseButton7");

var uploadbutton7 = document.getElementById('uploadButton7');
var seven = document.getElementById('7')

//add events to those 2 buttons
recordButton7.addEventListener("click", startRecording7);
stopButton7.addEventListener("click", stopRecording7);
pauseButton7.addEventListener("click", pauseRecording7);

var count = 0
function myFunction7() {
    if (recordButton7.style.display == "none" && stopButton7.style.display == "none" && pauseButton7.style.display == "none") {
        console.log("display three button")
        startrecordbtn7.style.display = "none"
        playpaussample7.style.display = "none"
        recordButton7.style.display = "block";
        stopButton7.style.display = "block";
        pauseButton7.style.display = "block";
    } else {
        recordButton7.style.display = "none";
        stopButton7.style.display = "none";
        pauseButton7.style.display = "none";
    }
}

function playpause7() {
    if (count == 0) {
        count = 1
        audio7.play();
        playpaussample7.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample7.innerHTML = "Play &#9658";
    }
}

function startRecording7() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton7.disabled = true;
    stopButton7.disabled = false;
    pauseButton7.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton7.disabled = false;
        stopButton7.disabled = true;
        pauseButton7.disabled = true
    });
}

function pauseRecording7() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton7.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton7.innerHTML = "Pause";

    }
}

function stopRecording7() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton7.disabled = true;
    recordButton7.disabled = false;
    pauseButton7.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton7.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton7.style.display = "none"
    pauseButton7.style.display = "none"
    stopButton7.style.display = "none"

    uploadbutton7.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink7);
}

function createDownloadLink7(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Vowel-a"

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton7.href = "#";
    uploadbutton7.innerHTML = "Upload";
    uploadbutton7.addEventListener("click", function (event) {
        seven.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);
    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton7)//add the upload link to li

    //add the li element to the ol
    recordingsList7.appendChild(li);

}

// ################################################################################################
// ################################################################################################
// ################################################################################################

var audio8 = document.getElementById("audio8")
var startrecordbtn8 = document.getElementById("startrecord8");
var playpaussample8 = document.getElementById("playpausebtn8")

var recordButton8 = document.getElementById("recordButton8");
var stopButton8 = document.getElementById("stopButton8");
var pauseButton8 = document.getElementById("pauseButton8");

var uploadbutton8 = document.getElementById('uploadButton8');
var eight = document.getElementById('8')

//add events to those 2 buttons
recordButton8.addEventListener("click", startRecording8);
stopButton8.addEventListener("click", stopRecording8);
pauseButton8.addEventListener("click", pauseRecording8);

var count = 0
function myFunction8() {
    if (recordButton8.style.display == "none" && stopButton8.style.display == "none" && pauseButton8.style.display == "none") {
        console.log("display three button")
        startrecordbtn8.style.display = "none"
        playpaussample8.style.display = "none"
        recordButton8.style.display = "block";
        stopButton8.style.display = "block";
        pauseButton8.style.display = "block";
    } else {
        recordButton8.style.display = "none";
        stopButton8.style.display = "none";
        pauseButton8.style.display = "none";
    }
}

function playpause8() {
    if (count == 0) {
        count = 1
        audio8.play();
        playpaussample8.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample8.innerHTML = "Play &#9658";
    }
}

function startRecording8() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton8.disabled = true;
    stopButton8.disabled = false;
    pauseButton8.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton8.disabled = false;
        stopButton8.disabled = true;
        pauseButton8.disabled = true
    });
}

function pauseRecording8() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton8.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton8.innerHTML = "Pause";

    }
}

function stopRecording8() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton8.disabled = true;
    recordButton8.disabled = false;
    pauseButton8.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton8.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton8.style.display = "none"
    pauseButton8.style.display = "none"
    stopButton8.style.display = "none"

    uploadbutton8.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink8);
}

function createDownloadLink8(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Vowel-e"
    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton8.href = "#";
    uploadbutton8.innerHTML = "Upload";
    uploadbutton8.addEventListener("click", function (event) {
        eight.style.display = 'block'
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);

        

    })
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton8)//add the upload link to li

    //add the li element to the ol
    recordingsList8.appendChild(li);
    
    
}

// ################################################################################################
// ################################################################################################
// ################################################################################################

var audio9 = document.getElementById("audio9")
var startrecordbtn9 = document.getElementById("startrecord9");
var playpaussample9 = document.getElementById("playpausebtn9")

var recordButton9 = document.getElementById("recordButton9");
var stopButton9 = document.getElementById("stopButton9");
var pauseButton9 = document.getElementById("pauseButton9");

var uploadbutton9 = document.getElementById('uploadButton9');
var nine = document.getElementById('9')

//add events to those 2 buttons
recordButton9.addEventListener("click", startRecording9);
stopButton9.addEventListener("click", stopRecording9);
pauseButton9.addEventListener("click", pauseRecording9);

var count = 0
function myFunction9() {
    if (recordButton9.style.display == "none" && stopButton9.style.display == "none" && pauseButton9.style.display == "none") {
        console.log("display three button")
        startrecordbtn9.style.display = "none"
        playpaussample9.style.display = "none"
        recordButton9.style.display = "block";
        stopButton9.style.display = "block";
        pauseButton9.style.display = "block";
    } else {
        recordButton9.style.display = "none";
        stopButton9.style.display = "none";
        pauseButton9.style.display = "none";
    }
}

function playpause9() {
    if (count == 0) {
        count = 1
        audio9.play();
        playpaussample9.innerHTML = "Pause &#9208";
    } else {
        count = 0;
        audio2.pause();
        playpaussample9.innerHTML = "Play &#9658";
    }
}

function startRecording9() {
    console.log("recordButton2 clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video: false }

    /*
        Disable the record button until we get a success or fail from getUserMedia() 
    */

    recordButton9.disabled = true;
    stopButton9.disabled = false;
    pauseButton9.disabled = false

    /*
        We're using the standard promise based getUserMedia() 
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format 
        document.getElementById("formats").innerHTML = "Format: 1 channel pcm @ " + audioContext.sampleRate / 1000 + "kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /* 
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input, { numChannels: 1 })

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function (err) {
        //enable the record button if getUserMedia() fails
        recordButton9.disabled = false;
        stopButton9.disabled = true;
        pauseButton9.disabled = true
    });
}

function pauseRecording9() {
    console.log("pauseButton2 clicked rec.recording=", rec.recording);
    if (rec.recording) {
        //pause
        rec.stop();
        pauseButton9.innerHTML = "Resume";
    } else {
        //resume
        rec.record()
        pauseButton9.innerHTML = "Pause";

    }
}

function stopRecording9() {
    console.log("stopButton2 clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton9.disabled = true;
    recordButton9.disabled = false;
    pauseButton9.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton9.innerHTML = "Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();
    recordButton9.style.display = "none"
    pauseButton9.style.display = "none"
    stopButton9.style.display = "none"

    uploadbutton9.style.display = 'block'

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink9);
}

function createDownloadLink9(blob) {

    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');

    //name of .wav file to use during upload and download (without extendion)
    // var filename = new Date().toISOString();
    var filename = "Vowel-o"

    //add controls to the <audio> element
    au.controls = true;
    au.src = url;

    
    //add the new audio element to li
    li.appendChild(au);

    uploadbutton9.href = "#";
    uploadbutton9.innerHTML = "Upload";
    uploadbutton9.addEventListener("click", function (event) {
        // nine.style.display = 'block'
        result.disabled = false
        var xhr = new XMLHttpRequest();
        xhr.onload = function (e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/upload_file", true);
        xhr.send(fd);
        
    }
    

    )
    li.appendChild(document.createTextNode(" "))//add a space in between
    li.appendChild(uploadbutton9)//add the upload link to li

    //add the li element to the ol
    
    recordingsList9.appendChild(li);
    
}

function getresult() 
{
    if (result.disabled == true){
        window.alert("Please first record all the files!!!")
    }
        
}