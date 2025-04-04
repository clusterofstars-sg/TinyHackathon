/**
 * Download Loading Overlay
 * 
 * This script handles showing and hiding the loading overlay when download links are clicked.
 * It shows a loading spinner when a user initiates a download and automatically hides it
 * when the page finishes loading after the download completes.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the loading overlay
    initDownloadOverlay();
});

/**
 * Initialize the download loading overlay functionality
 */
function initDownloadOverlay() {
    // Get all download links (both regular download and refresh download)
    const downloadLinks = document.querySelectorAll('a[href^="/download"]');
    
    // Add click event listener to each download link
    downloadLinks.forEach(link => {
        link.addEventListener('click', function(event) {
            // Show loading overlay
            showDownloadOverlay();
        });
    });

    // Hide overlay if we're on the download page (after a download completed)
    if (window.location.pathname.startsWith('/download')) {
        // Use setTimeout to ensure the page is fully loaded
        setTimeout(hideDownloadOverlay, 500);
    }
}

/**
 * Show the download loading overlay
 */
function showDownloadOverlay() {
    const overlay = document.getElementById('download-loading-overlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

/**
 * Hide the download loading overlay
 */
function hideDownloadOverlay() {
    const overlay = document.getElementById('download-loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Make hideDownloadOverlay available globally
window.hideDownloadOverlay = hideDownloadOverlay;
