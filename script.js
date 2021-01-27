function openCvReady() {
  var model = undefined;
  
  async function loadModel() {
    model = await tf.loadLayersModel('https://raw.githubusercontent.com/Raindeca/MyFacialApp/master/tfjs_files/model.json');
    console.log(model.summary());
    console.log('Model & Metadata Loaded Succesfully');
  }

  class L2 {
    static className = 'L2';
    constructor(config) {
      return tf.regularizers.l1l2(config);
    }
  }
  tf.serialization.registerClass(L2);



  cv['onRuntimeInitialized'] = () => {
    let video = document.getElementById('cam_input');
    // using WebRTC to get media stream
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
      })
      .catch(function (err) {
        console.log('An error has occured! ' + err);
      });let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let gray = new cv.Mat();
    let cap = new cv.VideoCapture(cam_input);
    let faces = new cv.RectVector();
    // let eyes = new cv.RectVector();
    let faceClassifier = new cv.CascadeClassifier();
    // let eyeClassifier = new cv.CascadeClassifier();
    let utils = new Utils('errorMessage');
    let faceCascade = 'haarcascade_frontalface_default.xml';
    model = loadModel();
    // let eyeCascade = 'haarcascade_eye.xml';
    utils.createFileFromUrl(faceCascade, faceCascade, () => {
      faceClassifier.load(faceCascade);
    });
    // utils.createFileFromUrl(eyeCascade, eyeCascade, () => {
    //   eyeClassifier.load(eyeCascade);
    // });
    const FPS = 30;
    async function frameRuntime() {
      cap.read(src);
      // let revFlatten = tf.tensor(src, [2,2]);
      // console.log(revFlatten);
      // console.log(src.data)

      // let temp = reverseCVFlat(src.data, 640);
      // console.log(temp);
      // return
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
      // let dst = new cv.Mat();
      try {
        faceClassifier.detectMultiScale(gray, faces, 1.1, 3, 0);
        for (let i = 0; i < faces.size(); ++i) {
          let face = faces.get(i);
          let point1 = new cv.Point(face.x, face.y);
          let point2 = new cv.Point(face.x + face.width, face.y + face.height);
          
          cv.rectangle(src, point1, point2, [255, 0, 0, 255]);

          let window = new cv.Rect(face.x, face.y, face.width, face.height)
          console.log(window)
          console.log(gray)
          let croppedFrame = gray.roi(window);
          cv.resize(croppedFrame, croppedFrame, new cv.Size(48, 48), cv.INTER_AREA)
          let rawInput = Array.from(croppedFrame.data).map(value => value / 255)

          const image = grouping(rawInput.map(value => [value]), 48)
          const input = [image]

          const RESULT_MAP = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

          console.log(input)

          const payload = model.predict(tf.tensor(input))
          const probas = eval(payload.toString().replace(/Tensor\s*/, ''))[0]

          function argmax(arr) {
            let maxIndex = -1
            let maxValue = Number.MIN_VALUE

            for (let i=0; i < arr.length; i++) {
              if (maxValue < arr[i]) {
                maxValue = arr[i]
                maxIndex = i
              }
            }

            return maxIndex
          }

          const index = argmax(probas)

          cv.putText(src, RESULT_MAP[index], {x: face.x, y: face.y}, cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 255, 0, 255], 2.0)
          // console.log(result)
          // let roi = cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
          // let dsize = new cv.Size(48, 48);
          // let resize = cv.resize(roi, dst, dsize, cv.INTER_AREA);
          // let imgArr = roi.clone();
          // let reshape = tf.expandDims(roi);
          // reshape = reshape / 255;

          // let predictions = model.predict(reshape);

          // let max_index = argMax(predictions[0]);
          
          // let emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];
          // let predicted_emotion = emotions[max_index];

          // cv.putText(cap.read(src), predicted_emotion, (face.x, face.y), cv.FONT_HERSHEY_SIMPLEX, 1.0, [0, 0, 255, 2]);

          // console.log(dst);
          // let dsize = new cv.Size(48, 48);
          // let dst = new cv.Mat();
          // let resize = cv.resize(src, dst, dsize, CV.INTER_AREA);
          // console.log(dst);
          
          // let resized = tf.image.resizeBilinear(roi_gray, [48,48]);
          // console.log(resized);


          // eyeClassifier.detectMultiScale(gray, eyes, 1.1, 3, 0);
          // for (let j = 0; j < eyes.size(); j++) {
          //   let eye = eyes.get(j);
          //   let point1 = new cv.Point(eye.x, eye.y);
          //   let point2 = new cv.Point(eye.x + eye.width, eye.y + eye.height);
          //   cv.rectangle(src, point1, point2, [255, 0, 0, 255]);
          // }
        }
      } catch (err) {
        console.log(err);
      }
      
      cv.imshow('canvas_output', src);
    }

    setInterval(async () => {
      setTimeout(frameRuntime, 0)
    }, 1000 / FPS)

    //ArgMax
    function argMax(array) {
      return [].reduce.call(array, (m, c, i, arr) => c > arr[m] ? i : m, 0)
    }
    
    // Grouping members
    function grouping(data, col) {
      let result = []
      let row = []
      for(let i = 0; i < data.length; i++) {
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
  }
}