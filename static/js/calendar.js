document.addEventListener('DOMContentLoaded', function() {
  var calendarEl = document.getElementById('calendar');
  var calendar = new FullCalendar.Calendar(calendarEl, {
    height: '390px',
    expandRows: true,
    headerToolbar: {
        left: 'prev',
        center: 'title',
        right: 'next'
    },
    initialView: 'dayGridMonth'
  });
  calendar.render();
});