// Wait until the DOM is ready.
$(document).ready(function(e) {
  
  // Add a click handler to the submit button.
  $("#guessButton").click(function(e) {
    
    // Prevent the form actually submitting.
    e.preventDefault();
    
    // Get the number of numbers to generate.
    noofnos = $("#noofnos").val();
    
    // Send AJAX request for new numbers.
    $.post("/image", {"noofnos": noofnos}, function(data){
      
      // Update the text area with the numbers.
      $("#randomNumbers").text(data.randomNos.join("\n"));
    
    });
  
  });

});