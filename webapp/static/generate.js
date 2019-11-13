let canvas = document.getElementById("digitCanvas");
let ctx = canvas.getContext("2d");
let painting = document.getElementById("content");
let paintStyle = getComputedStyle(painting);
canvas.width = parseInt(paintStyle.getPropertyValue("width"));
canvas.height = parseInt(paintStyle.getPropertyValue("height"));

let mouse = { x: 0, y: 0 };

canvas.addEventListener('mousemove', function (e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
}, false);

ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = '#ff0000';

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
    let imgURL = digitCanvas.toDataURL();
    console.log("generate.js imgURL  ", imgURL)
     
    // Prevent the form actually submitting.
    e.preventDefault();
    
    
    $.post("/guess", {"imgURL": imgURL}, function(data){
     
     
      $("#guess").text();
    
    });
  
  });                  

});

