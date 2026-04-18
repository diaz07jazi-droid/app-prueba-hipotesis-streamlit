import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from google import genai  # NUEVO

st.title("App de Prueba de Hipótesis")

st.write("Esta aplicación permite analizar distribuciones y realizar pruebas de hipótesis.")

# -------------------------
# FUNCIÓN PARA CONSULTAR GEMINI
# -------------------------
def analizar_con_ia(x_barra, mu, n_muestra, sigma, alpha, tipo, z, p_value, rechazo):
    try:
        # La API key se toma desde .streamlit/secrets.toml
        api_key = st.secrets["GEMINI_API_KEY"]

        client = genai.Client(api_key=api_key)

        decision_app = "Se rechaza H0" if rechazo else "No se rechaza H0"

        prompt = f"""
Eres un asistente estadístico. Analiza la siguiente prueba Z y responde en español claro, breve y académico.

Resumen estadístico:
- Media muestral: {x_barra:.4f}
- Media hipotética (H0): {mu:.4f}
- Tamaño de muestra: {n_muestra}
- Sigma poblacional conocida: {sigma:.4f}
- Nivel de significancia alpha: {alpha}
- Tipo de prueba: {tipo}
- Estadístico Z calculado: {z:.4f}
- p-value: {p_value:.6f}
- Decisión automática de la app: {decision_app}

Responde exactamente con esta estructura:
1. Decisión sobre H0
2. Explicación breve del resultado
3. Si los supuestos de la prueba Z parecen razonables
4. Si tu interpretación coincide o no con la decisión automática de la app
"""

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        return response.text

    except KeyError:
        return (
            "Error: No se encontró la API key. "
            "Agrega GEMINI_API_KEY en el archivo .streamlit/secrets.toml"
        )
    except Exception as e:
        return f"Error al conectar con la IA: {e}"


# -------------------------
# SELECCIÓN DE FUENTE DE DATOS
# -------------------------
st.header("Carga o generación de datos")

fuente_datos = st.radio(
    "Seleccione la fuente de los datos:",
    ["Generar datos sintéticos", "Cargar archivo CSV"]
)

datos = None
df = None
columna_seleccionada = None

# -------------------------
# DATOS SINTÉTICOS
# -------------------------
if fuente_datos == "Generar datos sintéticos":
    n = st.slider("Tamaño de muestra", 30, 200, 50)
    media = st.number_input("Media poblacional para generar datos", value=50.0)
    desv = st.number_input("Desviación estándar poblacional conocida (sigma)", value=10.0, min_value=0.1)

    datos = np.random.normal(media, desv, n)
    df = pd.DataFrame(datos, columns=["Datos"])
    columna_seleccionada = "Datos"

    st.write("Vista previa de los datos generados:")
    st.dataframe(df)

# -------------------------
# CARGA DE CSV
# -------------------------
elif fuente_datos == "Cargar archivo CSV":
    archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

    if archivo is not None:
        try:
            # Más tolerante con distintos separadores
            df = pd.read_csv(archivo, sep=None, engine="python")
            st.write("Vista previa del archivo cargado:")
            st.dataframe(df)

            columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()

            if len(columnas_numericas) > 0:
                columna_seleccionada = st.selectbox(
                    "Selecciona la columna numérica a analizar:",
                    columnas_numericas
                )

                datos = df[columna_seleccionada].dropna().values

                st.write(f"Se analizará la columna: **{columna_seleccionada}**")
                st.write(f"Número de observaciones válidas: **{len(datos)}**")

                if len(datos) < 30:
                    st.warning("La muestra tiene menos de 30 datos. La consigna pide n ≥ 30 para la prueba Z.")
            else:
                st.error("El archivo no contiene columnas numéricas.")
        except Exception as e:
            st.error(f"No se pudo leer el archivo CSV: {e}")
    else:
        st.info("Por favor, sube un archivo CSV para continuar.")

# Solo continuar si ya existen datos válidos
if datos is not None and len(datos) > 0:

    # Para la prueba Z se necesita sigma conocida
    st.header("Parámetros para la prueba")

    if fuente_datos == "Generar datos sintéticos":
        sigma = desv
        st.write(f"Sigma poblacional conocida usada: **{sigma:.4f}**")
    else:
        sigma = st.number_input(
            "Ingrese la desviación estándar poblacional conocida (sigma)",
            value=10.0,
            min_value=0.1
        )

    # -------------------------
    # GRÁFICAS
    # -------------------------
    st.header("Visualización de distribuciones")

    st.subheader("Histograma")
    fig, ax = plt.subplots()
    ax.hist(datos, bins=15, edgecolor="black", alpha=0.7)
    ax.set_title("Histograma de los datos")
    ax.set_xlabel("Valores")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    st.subheader("Boxplot")
    fig2, ax2 = plt.subplots()
    ax2.boxplot(datos, vert=True)
    ax2.set_title("Boxplot de los datos")
    st.pyplot(fig2)

    st.subheader("KDE")
    fig3, ax3 = plt.subplots()
    sns.kdeplot(datos, ax=ax3, fill=True)
    ax3.set_title("Curva de densidad KDE")
    ax3.set_xlabel("Valores")
    st.pyplot(fig3)

    # -------------------------
    # ANÁLISIS
    # -------------------------
    st.header("Análisis de la distribución")

    respuesta1 = st.selectbox("¿La distribución parece normal?", ["Sí", "No"])
    respuesta2 = st.selectbox("¿Hay sesgo?", ["Sí", "No"])
    respuesta3 = st.selectbox("¿Hay outliers?", ["Sí", "No"])

    st.write("**Tus respuestas:**")
    st.write(f"- Normalidad: {respuesta1}")
    st.write(f"- Sesgo: {respuesta2}")
    st.write(f"- Outliers: {respuesta3}")

    # -------------------------
    # PRUEBA DE HIPÓTESIS
    # -------------------------
    st.header("Prueba de Hipótesis Z")

    mu = st.number_input("Media hipotética (H0: μ = ...)", value=50.0)
    alpha = st.number_input("Nivel de significancia (α)", value=0.05, min_value=0.001, max_value=0.2, step=0.01)

    tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

    x_barra = np.mean(datos)
    n_muestra = len(datos)

    z = (x_barra - mu) / (sigma / np.sqrt(n_muestra))

    st.write(f"**Media muestral:** {x_barra:.4f}")
    st.write(f"**Tamaño de muestra:** {n_muestra}")
    st.write(f"**Sigma poblacional conocida:** {sigma:.4f}")
    st.write(f"**Estadístico Z calculado:** {z:.4f}")

    # -------------------------
    # p-value y región crítica
    # -------------------------
    if tipo == "Bilateral":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        z_crit_izq = stats.norm.ppf(alpha / 2)
        z_crit_der = stats.norm.ppf(1 - alpha / 2)
        region_texto = f"Rechazar H0 si Z < {z_crit_izq:.4f} o Z > {z_crit_der:.4f}"
        rechazo = z < z_crit_izq or z > z_crit_der

    elif tipo == "Cola izquierda":
        p_value = stats.norm.cdf(z)
        z_crit = stats.norm.ppf(alpha)
        region_texto = f"Rechazar H0 si Z < {z_crit:.4f}"
        rechazo = z < z_crit

    elif tipo == "Cola derecha":
        p_value = 1 - stats.norm.cdf(z)
        z_crit = stats.norm.ppf(1 - alpha)
        region_texto = f"Rechazar H0 si Z > {z_crit:.4f}"
        rechazo = z > z_crit

    st.write(f"**p-value:** {p_value:.6f}")
    st.write(f"**Región crítica:** {region_texto}")

    if rechazo:
        st.error("Decisión automática: Se rechaza H0")
    else:
        st.success("Decisión automática: No se rechaza H0")

    # -------------------------
    # INTERPRETACIÓN AUTOMÁTICA
    # -------------------------
    st.subheader("Interpretación automática")

    if rechazo:
        st.write(
            "Con base en el nivel de significancia seleccionado, existe evidencia estadística suficiente para rechazar la hipótesis nula."
        )
    else:
        st.write(
            "Con base en el nivel de significancia seleccionado, no existe evidencia estadística suficiente para rechazar la hipótesis nula."
        )

    # -------------------------
    # CURVA NORMAL + ZONA DE RECHAZO
    # -------------------------
    st.header("Curva normal con zona de rechazo")

    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.plot(x, y, color="blue", label="Distribución normal estándar")

    if tipo == "Bilateral":
        x_izq = np.linspace(-4, z_crit_izq, 300)
        y_izq = stats.norm.pdf(x_izq)
        ax4.fill_between(x_izq, y_izq, color="red", alpha=0.4, label="Zona de rechazo")

        x_der = np.linspace(z_crit_der, 4, 300)
        y_der = stats.norm.pdf(x_der)
        ax4.fill_between(x_der, y_der, color="red", alpha=0.4)

        x_centro = np.linspace(z_crit_izq, z_crit_der, 300)
        y_centro = stats.norm.pdf(x_centro)
        ax4.fill_between(x_centro, y_centro, color="green", alpha=0.3, label="Zona de no rechazo")

        ax4.axvline(z_crit_izq, color="black", linestyle="--", label=f"Z crítico izq = {z_crit_izq:.2f}")
        ax4.axvline(z_crit_der, color="black", linestyle="--", label=f"Z crítico der = {z_crit_der:.2f}")

    elif tipo == "Cola izquierda":
        x_rechazo = np.linspace(-4, z_crit, 300)
        y_rechazo = stats.norm.pdf(x_rechazo)
        ax4.fill_between(x_rechazo, y_rechazo, color="red", alpha=0.4, label="Zona de rechazo")

        x_no_rechazo = np.linspace(z_crit, 4, 300)
        y_no_rechazo = stats.norm.pdf(x_no_rechazo)
        ax4.fill_between(x_no_rechazo, y_no_rechazo, color="green", alpha=0.3, label="Zona de no rechazo")

        ax4.axvline(z_crit, color="black", linestyle="--", label=f"Z crítico = {z_crit:.2f}")

    elif tipo == "Cola derecha":
        x_rechazo = np.linspace(z_crit, 4, 300)
        y_rechazo = stats.norm.pdf(x_rechazo)
        ax4.fill_between(x_rechazo, y_rechazo, color="red", alpha=0.4, label="Zona de rechazo")

        x_no_rechazo = np.linspace(-4, z_crit, 300)
        y_no_rechazo = stats.norm.pdf(x_no_rechazo)
        ax4.fill_between(x_no_rechazo, y_no_rechazo, color="green", alpha=0.3, label="Zona de no rechazo")

        ax4.axvline(z_crit, color="black", linestyle="--", label=f"Z crítico = {z_crit:.2f}")

    ax4.axvline(z, color="purple", linestyle="-", linewidth=2, label=f"Z calculado = {z:.2f}")

    ax4.set_title("Curva normal estándar con región crítica")
    ax4.set_xlabel("Valores Z")
    ax4.set_ylabel("Densidad")
    ax4.legend()
    ax4.grid(alpha=0.3)

    st.pyplot(fig4)

    # -------------------------
    # IA REAL
    # -------------------------
    st.header("Asistente de IA")

    if st.button("Analizar resultados con IA"):
        with st.spinner("Consultando la IA..."):
            respuesta_ia = analizar_con_ia(
                x_barra=x_barra,
                mu=mu,
                n_muestra=n_muestra,
                sigma=sigma,
                alpha=alpha,
                tipo=tipo,
                z=z,
                p_value=p_value,
                rechazo=rechazo
            )

        st.subheader("Respuesta de la IA")
        st.write(respuesta_ia)

        st.subheader("Comparación con la decisión automática")
        decision_app = "Se rechaza H0" if rechazo else "No se rechaza H0"
        st.write(f"**Decisión de la app:** {decision_app}")
        st.write("**Compara esta decisión con la explicación de la IA y descríbelo en tu reporte.**")