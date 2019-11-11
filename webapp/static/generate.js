var canvas = document.getElementById("digitCanvas");
var ctx = canvas.getContext("2d");
var painting = document.getElementById("content");
var paintStyle = getComputedStyle(painting);
canvas.width = parseInt(paintStyle.getPropertyValue("width"));
canvas.height = parseInt(paintStyle.getPropertyValue("height"));

var mouse = { x: 0, y: 0 };

canvas.addEventListener('mousemove', function (e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
}, false);

ctx.lineWidth = 3;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#FF0000';

canvas.addEventListener('mousedown', function (e) {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
    canvas.addEventListener('mousemove', onPaint, false);
}, false);

canvas.addEventListener('mouseup', function () {
    canvas.removeEventListener('mousemove', onPaint, false);
}, false);


let onPaint = function () {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
};

function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

// Wait until the DOM is ready.
$(document).ready(function(e) {
  
  // Add a click handler to the submit button.
  $("#guessButton").click(function(e) {
    var imgURL = digitCanvas.toDataURL();
    console.log("get it ", imgURL)
     
    // Prevent the form actually submitting.
    e.preventDefault();
    
    // Send AJAX request for new numbers.
    $.post("/guess", {"imgURL": imgURL}, function(data){
     
      // Update the text area with the numbers.
      $("#guess").text();
    
    });
  
  });                  

});

