let model; 

async function learnLinear() {
    model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    const xs = tf.tensor2d([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [17, 1]);
    const ys = tf.tensor2d([-13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19], [17, 1]);

    const surface = { name: 'Loss', tab: 'Training' };
    const history = [];

    await model.fit(xs, ys, {
            epochs: 350, 
            callbacks : {
                onEpochEnd: (epoch, logs) => {
                    history.push(logs);
                    tfvis.show.history(surface, history, ['loss']);
                }
            }
         },  
    );

    // Muestra mensaje de entrenamiento finalizado
    Swal.fire({
        icon: 'success',
        title: 'Entrenamiento finalizado',
        text: 'El modelo está listo para usarse'
    });

    console.log("Entrenamiento finalizado. Modelo listo para usar.");
}

// Evento de clic en el botón "Entrenar Modelo"
document.getElementById('entrenar_modelo').addEventListener('click', async () => {
    await learnLinear();
});

// Evento de clic en el botón "Predecir"
document.getElementById('predict_button').addEventListener('click', async () => {
    if (!model) {
        // Muestra mensaje de error si el modelo no ha sido entrenado
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: 'Primero debes entrenar el modelo'
        });
        console.error("El modelo aún no está entrenado.");
        return; // Salir de la función si el modelo no está entrenado
    }

    const inputNumber = parseFloat(document.getElementById('input_number').value);
        // Valida que el valor de entrada no esté vacío
        if (isNaN(inputNumber)) {
            // Muestra mensaje de error si el valor está vacío
            Swal.fire({
                icon: 'error',
                title: 'Error',
                text: 'Ingresa un valor válido antes de realizar la predicción'
            });
            console.error("Valor de entrada vacío.");
            return; // Salir de la función si el valor está vacío
        }
    const prediction = model.predict(tf.tensor2d([inputNumber], [1, 1]));

    // Muestra el mensaje con el resultado de la predicción
    const outputMessage = `El resultado de predecir ${inputNumber} es: ${prediction.dataSync()[0]}`;
    document.getElementById('output_field').innerHTML = outputMessage;
});
