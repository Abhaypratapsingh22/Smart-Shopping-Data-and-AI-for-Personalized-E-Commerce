/**
 * Main JavaScript file for the E-Commerce AI Dashboard
 */

// Helper function to display error messages
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show';
    errorDiv.role = 'alert';
    errorDiv.innerHTML = `
        ${message}
        <button type="button" class="close" data-dismiss="alert" aria-label="Close">
            <span aria-hidden="true">&times;</span>
        </button>
    `;
    
    // Insert at the top of the content area
    const contentArea = document.querySelector('.container.mt-4');
    if (contentArea) {
        contentArea.insertBefore(errorDiv, contentArea.firstChild);
    }
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
}

// Format currency values
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 2
    }).format(value);
}

// Format percentage values
function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(value);
}

// Handle AJAX errors
function handleAjaxError(xhr, status, error) {
    let errorMessage = 'An error occurred while processing your request.';
    
    if (xhr.responseJSON && xhr.responseJSON.error) {
        errorMessage = xhr.responseJSON.error;
    } else if (error) {
        errorMessage += ' ' + error;
    }
    
    showError(errorMessage);
}

// Load a page with spinner
function loadPageWithSpinner(url, targetElement) {
    const spinner = `
        <div class="text-center my-5">
            <div class="spinner-border text-primary" role="status">
                <span class="sr-only">Loading...</span>
            </div>
            <p class="mt-2">Loading data...</p>
        </div>
    `;
    
    // Display spinner
    targetElement.innerHTML = spinner;
    
    // Load content
    fetch(url)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(html => {
            targetElement.innerHTML = html;
        })
        .catch(error => {
            showError('Failed to load content: ' + error.message);
        });
}

// Submit form with spinner and handle response
function submitFormWithSpinner(url, formData, successCallback) {
    // Show a spinner in the submit button
    const submitBtn = document.querySelector('button[type="submit"]');
    if (submitBtn) {
        const originalText = submitBtn.innerHTML;
        submitBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
            Processing...
        `;
        submitBtn.disabled = true;
    }
    
    // Submit the form
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Reset button
        if (submitBtn) {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
        
        // Handle success
        if (data.success) {
            successCallback(data);
        } else {
            showError(data.error || 'An error occurred while processing your request.');
        }
    })
    .catch(error => {
        // Reset button
        if (submitBtn) {
            submitBtn.innerHTML = originalText;
            submitBtn.disabled = false;
        }
        
        showError('Request failed: ' + error.message);
    });
}

// Document ready function
document.addEventListener('DOMContentLoaded', function() {
    // Add active class to current nav item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath) {
            link.classList.add('active');
        }
    });

    // Initialize any tooltips
    if (typeof $().tooltip === 'function') {
        $('[data-toggle="tooltip"]').tooltip();
    }
    
    // Initialize any popovers
    if (typeof $().popover === 'function') {
        $('[data-toggle="popover"]').popover();
    }
});