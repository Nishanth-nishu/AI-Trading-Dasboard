document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add active class to current nav item
    const currentPath = window.location.pathname;
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });

    // Currency pair selection checkboxes - ensure at least 2 are selected
    const analyzeForm = document.querySelector('form[action="/analyze"]');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', function(e) {
            const checkedBoxes = document.querySelectorAll('input[name="currency_pairs"]:checked');
            if (checkedBoxes.length < 2) {
                e.preventDefault();
                alert('Please select at least 2 currency pairs for analysis.');
            }
        });
    }

    // Chart initialization if we have chart data
    const chartContainers = document.querySelectorAll('.chart-container');
    if (chartContainers.length > 0) {
        chartContainers.forEach(container => {
            const pair = container.dataset.pair;
            const ctx = container.getContext('2d');
            
            // In a real implementation, you would fetch data from your Flask API endpoint
            // and use it to create the chart. This is just a placeholder.
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: `${pair} Price`,
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'top',
                        },
                        title: {
                            display: true,
                            text: `${pair} Price Chart`
                        }
                    }
                }
            });
        });
    }
});

// Function to refresh data (could be used with a refresh button)
function refreshData() {
    window.location.reload();
}