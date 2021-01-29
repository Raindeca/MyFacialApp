function openCvReady() {
  var model = undefined;
  const cameraSection = document.getElementById('cameraSection');
  const enableWebcamButton = document.getElementById('enableWebcam');
  const downloadButton = document.getElementById("downloadButton");

  // Load the tfjs model.
  async function loadModel() {
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/Raindeca/MyFacialApp/master/tfjs_files/model.json');
  }

  // Add support for regularizers on web
  class L2 {
    static className = 'L2';
    constructor(config) {
      return tf.regularizers.l1l2(config);
    }
  }
  tf.serialization.registerClass(L2);



  cv['onRuntimeInitialized'] = () => {

    // Using function to download result later
    function download() {
      var download = document.getElementById("download");
      var image = document.getElementById("canvas_output").toDataURL("image/png")
        .replace("image/png", "image/octet-stream");
      download.setAttribute("href", image);
    }

    downloadButton.addEventListener('click', download);


    let video = document.getElementById('cam_input');

    // using WebRTC to get media stream
    function getUserMediaSupported() {
      return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
    }

    if (getUserMediaSupported()) {
      enableWebcamButton.addEventListener('click', enableCam);
    } else {
      console.warn('getUserMedia() is not supported by your browser');
    }
    function enableCam(event) {

      cameraSection.classList.remove('invisible')
      navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function (stream) {
          video.srcObject = stream;
          video.play();
        })
        .catch(function (err) {
          console.log('An error has occured! ' + err);
        });

      let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
      let gray = new cv.Mat();
      let cap = new cv.VideoCapture(cam_input);
      let faces = new cv.RectVector();
      let faceClassifier = new cv.CascadeClassifier();
      let utils = new Utils('errorMessage');
      let faceCascade = 'haarcascade_frontalface_default.xml';

      model = loadModel();
      utils.createFileFromUrl(faceCascade, faceCascade, () => {
        faceClassifier.load(faceCascade);
      });


      const FPS = 30;
      async function frameRuntime() {
        cap.read(src);
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
        try {
          faceClassifier.detectMultiScale(gray, faces, 1.1, 3, 0);
          for (let i = 0; i < faces.size(); ++i) {
            let face = faces.get(i);
            let point1 = new cv.Point(face.x, face.y);
            let point2 = new cv.Point(face.x + face.width, face.y + face.height);

            cv.rectangle(src, point1, point2, [255, 0, 0, 255]);

            let window = new cv.Rect(face.x, face.y, face.width, face.height)
            let croppedFrame = gray.roi(window);
            cv.resize(croppedFrame, croppedFrame, new cv.Size(48, 48), cv.INTER_AREA)
            let rawInput = Array.from(croppedFrame.data).map(value => value / 255)

            const image = grouping(rawInput.map(value => [value]), 48)
            const input = [image]

            const RESULT_MAP = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            const payload = model.predict(tf.tensor(input))
            const probas = eval(payload.toString().replace(/Tensor\s*/, ''))[0]


            const index = argmax(probas)

            cv.putText(src, RESULT_MAP[index], { x: face.x, y: face.y }, cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0, 255], 2.0)

          }
        } catch (err) {
          console.log(err);
        }

        cv.imshow('canvas_output', src);

      }

      setInterval(async () => {
        setTimeout(frameRuntime, 0)
      }, 1000 / FPS)
    }


    // Grouping members
    function grouping(data, col) {
      let result = []
      let row = []
      for (let i = 0; i < data.length; i++) {
        row = [
          ...row,
          data[i]
        ]

        if ((i + 1) % col === 0) {
          result = [
            ...result,
            row,
          ]
          row = []
        }
      }
      return result
    }

    function argmax(arr) {
      let maxIndex = -1
      let maxValue = Number.MIN_VALUE

      for (let i = 0; i < arr.length; i++) {
        if (maxValue < arr[i]) {
          maxValue = arr[i]
          maxIndex = i
        }
      }

      return maxIndex
    }



  }

}