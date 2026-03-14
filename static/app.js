document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    
    // UI State Elements
    const statusIdle = document.getElementById('status-idle');
    const statusLoading = document.getElementById('status-loading');
    const statusSuccess = document.getElementById('status-success');
    const statusError = document.getElementById('status-error');
    
    // Dynamic Elements
    const salaryOutput = document.getElementById('salary-output');
    const errorMessage = document.getElementById('error-message');
    const resetBtn = document.getElementById('reset-btn');
    const retryBtn = document.getElementById('retry-btn');

    // Helper to switch right-panel states
    function setState(stateElement) {
        [statusIdle, statusLoading, statusSuccess, statusError].forEach(el => {
            el.classList.remove('active');
        });
        stateElement.classList.add('active');
    }

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Switch to Loading UI
        setState(statusLoading);
        
        // Gather JSON payload
        // Group all required form inputs dynamically into a payload
        const formData = new FormData(form);
        const payload = {};
        
        for (let [key, value] of formData.entries()) {
            // Try converting numbers, fallback to string if NaN
            const numVal = Number(value);
            payload[key] = isNaN(numVal) ? value : numVal;
        }

        try {
            // Delay slightly for smooth animation feel
            await new Promise(r => setTimeout(r, 600));

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle API HTTP Exceptions (e.g. model missing)
                throw new Error(data.detail || 'Failed to process prediction');
            }

            // Success Transition
            salaryOutput.textContent = data.formatted_salary;
            setState(statusSuccess);

        } catch (error) {
            errorMessage.textContent = error.message;
            setState(statusError);
        }
    });

    // Reset Buttons Behavior
    resetBtn.addEventListener('click', () => setState(statusIdle));
    retryBtn.addEventListener('click', () => setState(statusIdle));
});
