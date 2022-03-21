function getFileName() {
    var x = document.getElementById('uploadfile')
    var y = document.getElementById('face_uploadfile')
    y.style.visibility = 'collapse'
    x.style.visibility = 'collapse'
    document.getElementById('fileName').innerHTML = x.value.split('\\').pop()
    document.getElementById('face_fileName').innerHTML = y.value.split('\\').pop()
}