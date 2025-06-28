document.addEventListener('DOMContentLoaded', function() {
    const resetButton = document.getElementById('resetButton');
    const nextStepButton = document.getElementById('nextStepButton'); // Nuevo botón

    const algorithms = ['A*', 'BFS', 'DFS', 'GA'];

    function updateUI(data) {
        algorithms.forEach(algo => {
            const safeAlgoId = algo.replace('*', 'Star'); // Para IDs HTML válidos
            
            const snakeImage = document.getElementById(`snakeImage${safeAlgoId}`);
            const timeSpan = document.getElementById(`time${safeAlgoId}`);
            const stepsSpan = document.getElementById(`steps${safeAlgoId}`);
            const gameOverSpan = document.getElementById(`gameOver${safeAlgoId}`);
            const applesEatenSpan = document.getElementById(`applesEaten${safeAlgoId}`);

            if (data[algo]) {
                snakeImage.src = `data:image/png;base64,${data[algo].img_data}`;
                timeSpan.textContent = `${data[algo].time}s`;
                stepsSpan.textContent = data[algo].steps;
                applesEatenSpan.textContent = `${data[algo].apples_eaten}/5`;
                gameOverSpan.textContent = data[algo].game_over ? '¡Game Over!' : 'Jugando';

                if (algo === 'A*') {
                    const openNodesSpan = document.getElementById(`openNodes${safeAlgoId}`);
                    const closedNodesSpan = document.getElementById(`closedNodes${safeAlgoId}`);
                    if (openNodesSpan && closedNodesSpan) {
                        openNodesSpan.textContent = data[algo].open_nodes.length;
                        closedNodesSpan.textContent = data[algo].closed_nodes.length;
                    }
                }
            } else {
                console.warn(`No data for algorithm: ${algo}`);
            }
        });
    }

    async function fetchGameState(endpoint) {
        try {
            const response = await fetch(endpoint);
            if (!response.ok) {
                const errorText = await response.text(); 
                throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error("Error fetching game state:", error);
            alert("Ocurrió un error al cargar el estado del juego. Revisa la consola para más detalles.");
            return null;
        }
    }

    // Función para manejar el "siguiente paso"
    async function nextStep() {
        const data = await fetchGameState('/step');
        if (data) {
            updateUI(data);
        }
    }

    // Función para manejar el reinicio
    async function resetGame() {
        const data = await fetchGameState('/reset');
        if (data) {
            updateUI(data);
        }
    }

    // Event listeners para los botones
    resetButton.addEventListener('click', resetGame);
    nextStepButton.addEventListener('click', nextStep); // Listener para el nuevo botón

    // Iniciar el juego automáticamente al cargar la página
    resetGame(); 
});