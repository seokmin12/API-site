$(document).ready(function() {
    // 감정분석 결과 출력
    var sentiment_result_div = document.getElementById('sentiment_result');
    var sentiment_predict = new URLSearchParams(location.search).get('sentiment_predict');
    sentiment_result_div.innerText = sentiment_predict;

    // 얼굴 인식 결과 출력
    var face_recognition_result_div = document.getElementById('face_result');
    var face_recognition_result = new URLSearchParams(location.search).get('cropped');
    if (face_recognition_result != null) {
        face_recognition_result_div.innerHTML = `<img src="{{ url_for('static', filename='uploadimage/face_crop.jpeg') }}" height="200" width="200">`;
    }

})
function getFileName() {
    var x = document.getElementById('uploadfile')
    var y = document.getElementById('face_uploadfile')
    y.style.visibility = 'collapse'
    x.style.visibility = 'collapse'
    document.getElementById('fileName').innerHTML = x.value.split('\\').pop()
    document.getElementById('face_fileName').innerHTML = y.value.split('\\').pop()
}