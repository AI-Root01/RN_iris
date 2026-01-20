# 🌸 Iris Classifier (TensorFlow) — MLP + MinMaxScaler

Proyecto en Python que entrena una red neuronal (MLP) con TensorFlow/Keras para clasificar flores del dataset **Iris** en 3 clases:

- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

El entrenamiento se hace a partir de archivos `.txt` por clase, se normalizan las features con **MinMaxScaler** y se guarda el modelo entrenado en formato `.h5`.

## ✨ Qué incluye

- 📥 Carga de datos desde 3 archivos (`Iris_setosa.txt`, `Iris_versicolor.txt`, `Iris_virginica.txt`)
- 🧹 Conversión de etiquetas a valores numéricos (0, 1, 2)
- 📏 Normalización con `MinMaxScaler`
- 🧠 Modelo MLP (Dense/ReLU) + salida Softmax
- 🧪 Evaluación con conjunto de prueba (accuracy/loss)
- 📈 Gráfica de pérdida vs épocas
- 💾 Guardado del modelo entrenado (`.h5`)

## 🧰 Requisitos

- Python 3.8+ (recomendado)
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- (Opcional) seaborn

Instalación rápida:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

> 📝 Si se ejecuta en Google Colab, normalmente TensorFlow ya está disponible.

## 📁 Dataset / Formato de archivos

El script espera **tres archivos** con el mismo formato CSV (con encabezado), uno por clase.

Cada fila debe tener:

- 4 features numéricas (float): `sepal_length, sepal_width, petal_length, petal_width`
- 1 etiqueta (string): `Iris-setosa`, `Iris-versicolor` o `Iris-virginica`

Ejemplo de fila:

```text
5.1,3.5,1.4,0.2,Iris-setosa
```

## ⚙️ Configuración de rutas (Colab)

Actualmente las rutas están definidas para Google Drive en Colab:

- Dataset:
  - `/content/drive/MyDrive/iris/iris_data/Iris_setosa.txt`
  - `/content/drive/MyDrive/iris/iris_data/Iris_versicolor.txt`
  - `/content/drive/MyDrive/iris/iris_data/Iris_virginica.txt`
- Guardado del modelo:
  - `/content/drive/MyDrive/modelosYpesos/iris_normal.h5`

Si lo vas a correr local, cambia estas rutas por rutas relativas del proyecto, por ejemplo:

```python
np.loadtxt("data/Iris_setosa.txt", ...)
model.save("models/iris_normal.h5")
```

## 🧠 Arquitectura del modelo

- Input: 4 features
- Dense(128) + ReLU
- Dense(64) + ReLU
- Dense(3) + Softmax

Compilación:

- Optimizer: Adam (`lr=0.001`)
- Loss: `sparse_categorical_crossentropy`
- Métrica: `accuracy`

## 🚀 Entrenamiento

El script entrena por default con:

- `epochs = 700`
- `batch_size = 64`
- `validation_split = 0.2` (tomado del set de entrenamiento)

Puedes cambiar `num_epochs` para ajustar tiempo/calidad:

```python
num_epochs = 200
```

## 🧪 Evaluación

Se realiza un split estratificado:

- `test_size = 0.2`
- `random_state = 42`

Y se imprime:

- `Loss`
- `Accuracy`

> 💡 El script también importa `confusion_matrix` y `classification_report`, pero no los usa actualmente. Puedes agregarlos para un reporte más completo.

## 📊 Visualización

Se grafica el historial de entrenamiento:

- Pérdida (`loss`) vs Época

Si quieres graficar también `accuracy`, puedes agregar:

```python
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.show()
```

## 💾 Output

- Modelo entrenado:
  - `iris_normal.h5`

## 🛟 Troubleshooting

- **Error de rutas**: confirma que montaste Drive (`from google.colab import drive; drive.mount('/content/drive')`) o ajusta a rutas locales.
- **`ValueError` al cargar datos**: revisa `delimiter=","`, encabezado (`skiprows=1`) y que las 5 columnas existan.
- **Overfitting**: 700 épocas puede ser demasiado; baja épocas, agrega regularización o early stopping.

## 📄 Licencia

Agrega una licencia (ej. MIT) si lo vas a publicar como open-source.
