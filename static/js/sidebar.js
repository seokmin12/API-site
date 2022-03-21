$(document).ready(function() {
    var sidebar = $('#sidebar');
    var main = $('.main');
    $('#sidebar_close_btn').click(function() {
        sidebar.removeClass("active")
        main.css('margin-left', '0');
    })
    $('#sidebar_open_btn').click(function() {
        sidebar.addClass("active");
        main.css('margin-left', '300px');
    })

    var dashboard_boolen = true
    $('#dashboards-btn').click(function() {
        if (dashboard_boolen == false) {
            $('#dashboards-menu').slideDown();
            $('#dashboard_angle').css({'transform': 'rotate(90deg)'});
            dashboard_boolen = true
        } else {
            $('#dashboards-menu').slideUp();
            $('#dashboard_angle').css({'transform': 'rotate(0deg)'});
            dashboard_boolen = false
        }
    })

    var document_boolen = true
    $('#document-btn').click(function() {
        if (document_boolen == false) {
            $('#document-menu').slideDown();
            $('#document_angle').css({'transform': 'rotate(90deg)'});
            document_boolen = true
        } else {
            $('#document-menu').slideUp();
            $('#document_angle').css({'transform': 'rotate(0deg)'});
            document_boolen = false
        }
    })
})