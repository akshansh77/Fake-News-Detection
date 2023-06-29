// document.getElementById('news_text').addEventListener('submit', function(event) {
//     event.preventDefault();

//     var formData = new FormData(this);
//     var data = {};
//     formData.forEach(function(value, key) {
//         data[key] = value;
//     });
//     console.log("ayya yahe pe")
//     var request = new XMLHttpRequest();
//     request.open('POST', 'http://127.0.0.1:5000/predict', true);
//     request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
//     request.onreadystatechange = function() {
//         if (request.readyState === 4 && request.status === 200) {
//             var response = JSON.parse(request.responseText);
//             var resultDiv = document.getElementById('result');
//             resultDiv.textContent = 'Response: ' + response.message;
//         }
//     };
//     request.send(new URLSearchParams(formData).toString());
// });

// document.getElementById('news_text').addEventListener('submit', function(event) {
//     event.preventDefault();

//     var formData = new FormData(this);
//     console.log("ayya yahe pe")
//     var request = new XMLHttpRequest();
//     request.open('POST', 'http://127.0.0.1:5000/predict', true);
//     request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
//     request.onreadystatechange = function() {
//         if (request.readyState === 4 && request.status === 200) {
//             var response = JSON.parse(request.responseText);
//             var resultDiv = document.getElementById('result');
//             resultDiv.textContent = 'Prediction: ' + response.prediction;
//         }
//     };
//     request.send(new URLSearchParams(formData).toString());
// })

// document.getElementById('news_text').addEventListener('submit', function(event) {
//     event.preventDefault();

//     var formData = new FormData(this);
//     console.log("ayya yahe pe")
//     var request = new XMLHttpRequest();
//     request.open('POST', 'http://127.0.0.1:5000/predict', true);
//     request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
//     request.onreadystatechange = function() {
//         if (request.readyState === 4 && request.status === 200) {
//             var response = JSON.parse(request.responseText);
//             var resultDiv = document.getElementById('result');
//             if (response.prediction === 0) {
//                 resultDiv.textContent = 'True';
//                 resultDiv.style.color = 'green';
//             } else if (response.prediction === 1) {
//                 resultDiv.textContent = 'False ';
//                 resultDiv.style.color = 'red';

//             }
//         }
//     };
//     request.send(new URLSearchParams(formData).toString());
// });

// $(document).ready(function() {
//     $('#predict-button').click(function() {
//         var text = $('#text-input').val();
//         if (text !== '') {
//             $('#prediction-result').text('Loading...');
//             $('#error-message').text('');
//             $.ajax({
//                 url: 'http://localhost:5000/predict',
//                 type: 'POST',
//                 data: {text: text},
//                 dataType: 'json',
//                 success: function(response) {
//                     if ('prediction' in response) {
//                         var prediction = response.prediction;
//                         $('#prediction-result').text('Prediction: ' + prediction);
//                     } else if ('error' in response) {
//                         var errorMessage = response.error;
//                         $('#error-message').text(errorMessage);
//                     }
//                 },
//                 error: function(xhr, status, error) {
//                     $('#prediction-result').text('');
//                     $('#error-message').text('An error occurred during prediction: ' + error);
//                 }
//             });
//         } else {
//             $('#prediction-result').text('');
//             $('#error-message').text('Please enter some text.');
//         }
//     });
// });

document.addEventListener('DOMContentLoaded', function () {
    var predictionForm = document.getElementById('prediction-form');
    var textInput = document.getElementById('text-input');
    var predictionResult = document.getElementById('prediction-result');

    predictionForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var text = textInput.value;
      if (text.trim() !== '') {
        makePrediction(text);
      }
    });

    function makePrediction(text) {
      var xhr = new XMLHttpRequest();
      xhr.open('POST', 'http://localhost:5000/predict', true);
      xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
          var response = JSON.parse(xhr.responseText);
          if ('prediction' in response) {

              if(response.prediction==1){
                ans='real'
              }else{
                ans='fake'
              }


            predictionResult.innerHTML = 'Prediction: ' + ans;
          } else if ('error' in response) {
            predictionResult.innerHTML = 'Error: ' + response.error;
          }
        }
      };
      xhr.send('text=' + encodeURIComponent(text));
    }
  });



