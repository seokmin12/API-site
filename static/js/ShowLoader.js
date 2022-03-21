$(document).ready(function() {
    $('#NL_btn').click(function() {
        if ($('#news_input').val() == '') {
            alert('기사 제목을 입력해 주세요.');
            return false;
        } else {
            $(this).css('display', 'none');
            $('#NL_loader').css('display', 'inline-block');
        }
    })
    $('#face_btn').click(function() {
        if ($('#face_uploadfile').val() == '') {
            alert('사진을 업로드해 주세요.');
            return false;
        } else {
            $(this).css('display', 'none');
            $('#face_loader').css('display', 'inline-block');
        }
    })
    $('#stock_predict_btn').click(function() {
        if ($('#stock_symbol').val() == '') {
            alert('종목 코드를 입력해 주세요.');
            return false;
        } else {
            $(this).css('display', 'none');
            $('#stock_loader').css('display', 'inline-block');
        }
    })
})
function InitialShowLoader() {
    document.getElementById('login').style.display = "none";
    document.getElementById('initial_loader').style.display = "inline-block";
}