document.addEventListener('DOMContentLoaded', () => {

    // === Image Preview ===
    const fileInput = document.querySelector('input[name="image"]');
    const preview = document.getElementById('imagePreview');

    if (fileInput && preview) {
        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            if (file) {
                if (preview.src) URL.revokeObjectURL(preview.src);
                preview.src = URL.createObjectURL(file);
                preview.classList.remove('hidden');
            }
        });
    }

    // === Ask ECHO Interaction ===
    const askForm = document.getElementById('echo-form');
    const echoInput = document.getElementById('echoQuestion');
    const echoOutput = document.getElementById('echoResult');

    if (askForm && echoInput && echoOutput) {
        askForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const question = echoInput.value.trim();
            if (!question) return;

            echoOutput.innerHTML = "<em>ECHO is reflecting on your question...</em>";

            try {
                const res = await fetch("/ask-echo", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ question })
                });

                const data = await res.json();
                echoOutput.innerHTML = `<blockquote>${data.response}</blockquote>`;
            } catch (err) {
                echoOutput.innerHTML = "<span class='error'>Something went wrong. Please try again.</span>";
            }
        });
    }

});
