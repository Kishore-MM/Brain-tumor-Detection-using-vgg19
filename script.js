document.addEventListener("DOMContentLoaded", function() {
    // Load Header Partial
    fetch("/static/partials/header.html")
      .then(response => response.text())
      .then(data => {
        document.getElementById("header-container").innerHTML = data;
      })
      .catch(err => console.error("Failed to load header:", err));
  
    // Load Footer Partial
    fetch("/static/partials/footer.html")
      .then(response => response.text())
      .then(data => {
        document.getElementById("footer-container").innerHTML = data;
      })
      .catch(err => console.error("Failed to load footer:", err));
  });
  