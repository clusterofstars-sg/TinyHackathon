/**
 * Submission page functionality
 */
document.addEventListener('DOMContentLoaded', function () {
    const storyBadges = document.querySelectorAll('.story-badge');

    // Add highlighting for the active story badge based on the current item
    const highlightCurrentItem = () => {
        const url = new URL(window.location.href);
        const currentItem = url.searchParams.get('item_id');

        // If no item_id in URL, try to get the first badge as default
        let defaultItemId = null;
        if (storyBadges.length > 0) {
            defaultItemId = storyBadges[0].getAttribute('data-id');
        }
        
        const targetItemId = currentItem || defaultItemId;

        if (targetItemId) {
            storyBadges.forEach(badge => {
                const itemId = badge.getAttribute('data-id');

                if (itemId === targetItemId) {
                    badge.classList.add('active');
                    badge.classList.add('bg-primary'); // Ensure active class has primary background
                } else {
                    badge.classList.remove('active');
                    badge.classList.remove('bg-primary');
                }
            });
        }
    };

    highlightCurrentItem();

    // Add keyboard navigation for prev/next item
    document.addEventListener('keydown', function (e) {
        // Left arrow key
        if (e.key === 'ArrowLeft') {
            const prevLink = document.querySelector('a.btn-outline-primary:not(.disabled) .fa-chevron-left');
            if (prevLink) prevLink.closest('a').click();
        }
        // Right arrow key
        else if (e.key === 'ArrowRight') {
            const nextLink = document.querySelector('a.btn-outline-primary:not(.disabled) .fa-chevron-right');
            if (nextLink) nextLink.closest('a').click();
        }
    });
});