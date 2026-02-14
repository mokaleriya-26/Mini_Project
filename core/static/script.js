// your_app/static/js/script.js

// Using window.onload ensures all elements are fully loaded before execution
window.onload = function() {
    
    // Check if the container exists
    const slideshowContainer = document.querySelector('.slideshow-container');
    if (!slideshowContainer) {
        console.error('Slideshow container not found!');
        return; 
    }
    
    // Configuration
    const slides = document.querySelectorAll('.hero-banner');
    const totalSlides = slides.length; 
    const slideWidth = 100; // Represents 100vw
    let currentSlide = 0;

    function nextSlide() {
        // Increment the slide index, wrapping around to the first slide (0)
        currentSlide = (currentSlide + 1) % totalSlides;
        
        // Calculate the translation offset in 'vw' units
        const offset = -currentSlide * slideWidth;
        
        // Apply the movement
        slideshowContainer.style.transform = `translateX(${offset}vw)`;
    }

    // Start the automatic slideshow timer: advances every 5 seconds (5000ms)
    // We delay the start slightly to ensure the initial transform is stable
    setTimeout(() => {
        setInterval(nextSlide, 5000); 
    }, 100); 
};