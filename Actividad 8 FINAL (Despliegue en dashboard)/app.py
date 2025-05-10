import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.figure_factory as ff
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import train_test_split
from PIL import Image


####DATAFRAME#####################################################################
@st.cache_data
def load_data():
    df = pd.read_csv("Datos_limpios_chicago.csv", index_col="id")
    df = df.drop(columns=["Unnamed: 0.1"])
    numeric_df = df.select_dtypes(include=['float', 'int'])
    numeric_cols = numeric_df.columns
    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns
    return df, numeric_df, numeric_cols, text_df, text_cols
####################################################################################

#########ETAPA 1###################################################################
def univariado(df):

    with st.sidebar:
        #VARIABLES CATEGORICAS IMPORTANTES
        col_categorica = st.selectbox("Variable categórica", options=[
                "room_type", "host_verifications", "host_response_time",
                "host_is_superhost", "property_type"
            ])
        show_pie = st.checkbox("Mostrar gráfico de pastel")

    st.header("Análisis univariado de variables categóricas")
    conteo = df[col_categorica].value_counts().reset_index()
    conteo.columns = [col_categorica, "frecuencia"]
    st.subheader("📊*Gráfico de barras*")
    st.plotly_chart(px.bar(conteo, x=col_categorica, y='frecuencia'))

    if show_pie:
        st.subheader("📊*Gráfico de pastel*")
        st.plotly_chart(px.pie(conteo, names=col_categorica, values='frecuencia'))
    st.subheader("📋*Tabla de frecuencia de la variable*")
    st.dataframe(conteo)
######################################################################################################

############ETAPA 2##################################################################################
def regresion( df, numeric_df, numeric_cols, text_df, text_cols):
    
    st.subheader("Análisis de regresión")
    with st.expander("🗒️ Instrucciones", expanded=True):
        st.markdown("""Antes de hacer una regresión, puedes generar un mapa de calor para observar las correlaciones entre variables.
                Despues selecciona el tipo de regresión que deseas aplicar, luego elige tus variables y finalmente haz clic en el botón **EJECUTAR MODELO** para generar la información y los gráficos.""")

    #MAPA DE CALOR
    st.markdown("### Mapa de calor de variables")
    numeric_df = numeric_df.loc[:, ~numeric_df.columns.duplicated()]
    selected_cols = st.multiselect('Selecciona variables para el análisis de correlación', numeric_df.columns)

    if selected_cols:
        corr_matrix = numeric_df[selected_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            labels={'color': 'Correlación'},
            title="Matriz de Correlación"
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        tipo_modelo = st.radio("Tipo de regresión", ["Regresión lineal simple", "Regresión lineal múltiple", 
            "Regresión logística"])
        #RLS EN SIDEBAR
        if tipo_modelo == "Regresión lineal simple":
            var_x = st.selectbox("Variable X", options=numeric_cols, key="rls_x")
            var_y = st.selectbox("Variable Y", options=numeric_cols, key="rls_y")
            ejecutar_simple = st.button("Ejecutar modelo simple")
        #RLM EN SIDEBAR
        elif tipo_modelo == "Regresión lineal múltiple":
            var_y = st.selectbox("Variable Y", options=numeric_cols, key="rlm_y")
            sugeridas = [col for col in selected_cols if col != var_y]
            var_xs = st.multiselect("Variables X", options=[col for col in numeric_cols if col != var_y], default=sugeridas)
            ejecutar_multiple = st.button("Ejecutar modelo múltiple")
        #RL EN SIDEBAR
        elif tipo_modelo == "Regresión logística":
            binarias = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() == 2]
            var_y = st.selectbox("Variable binaria", options=binarias)
            var_x = st.selectbox("Variable numérica", options=numeric_cols, key="rlg_x")
            ejecutar_logistica = st.button("Ejecutar modelo logístico")
    
    #MOSTRAR INFORMACIÓN DE RLS
    if tipo_modelo == "Regresión lineal simple" and 'ejecutar_simple' in locals() and ejecutar_simple:
        if var_x == var_y:
            st.warning("⚠️ Las variables X y Y no deben ser iguales en regresión lineal simple.")
            return
            
        data = df[[var_x, var_y]].dropna()
        X = data[[var_x]]
        y = data[var_y]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        correlacion = np.corrcoef(y, y_pred)[0, 1]
        st.success("✅ Modelo ejecutado correctamente.") 

        col1, col2, col3 = st.columns(3)

        with col1:
            #COEF CORRELACIÓN
            st.metric("Coeficiente de determinación", f"{r2_score(y, y_pred):.4f}")
        with col2:
            #COEF REGRESIÓN
            st.metric("Coeficiente de regresión", f"{model.coef_[0]:.4f}")
        with col3:
            #COEF CORRELACIÓN
            st.metric("Coeficiente de correlación", f"{correlacion:.4f}")

        #GRAFICO DE RLS
        st.plotly_chart(px.scatter(data, x=var_x, y=var_y, trendline="ols"))

    #MOSTRAR INFORMACIÓN DE RLM
    elif tipo_modelo == "Regresión lineal múltiple" and 'ejecutar_multiple' in locals() and ejecutar_multiple:
        if not var_xs:
            st.warning("⚠️ Selecciona al menos una variable X para continuar.")
            return
        data = df[[var_y] + var_xs].dropna()
        X = data[var_xs]
        y = data[var_y]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        correlacion = np.corrcoef(y, y_pred)[0, 1]
        st.success("✅ Modelo ejecutado correctamente.") 
       
        col1, col2 = st.columns(2)

        with col1:
            #COEF DETERMINACIÓN
            st.metric(label="Coeficiente de determinación", value=f"{r2_score(y, y_pred):.4f}")
        with col2:
            #COEF CORRELACIÓN
            st.metric(label="Coeficiente de correlación", value=f"{correlacion:.4f}")
        #COEF DE CADA VARIABLE
        st.markdown("### Coeficientes por variable")
        for var, coef in zip(var_xs, model.coef_):
            st.write(f"- **{var}**: {coef:.4f}")


        #GRAFICO DE RLM
        grafico = px.scatter(
            x=y,
            y=y_pred,
            labels={"x": "Valor real", "y": "Valor predictivo"},
        )
        st.plotly_chart(grafico)

    #MOSTRAR INFORMACIÓN DE RL
    elif tipo_modelo == "Regresión logística" and 'ejecutar_logistica' in locals() and ejecutar_logistica:
        data = df[[var_y, var_x]].dropna()
        data[var_y] = data[var_y].map({'t': 1, 'f': 0})
        X = data[[var_x]]
        y = data[var_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        model = LogisticRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        #SENSIBILIDAD
        recall_positivos = recall_score(y_test, y_pred, pos_label=1)
        recall_negativos = recall_score(y_test, y_pred, pos_label=0)
        st.success("✅ Modelo ejecutado correctamente.") 

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Exactitud total del modelo", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Sensibilida positiva (clase 1)", f"{recall_positivos:.4f}")
        with col3:
            st.metric("Sensibilidad negativa (clase 0)", f"{recall_negativos:.4f}")

        #MATRIZ DE CONFUSIÓN
        cm = confusion_matrix(y_test, y_pred)
        st.write("📌 Matriz de Confusión:")
        z = cm.tolist()
        x = ["Predicción 0", "Predicción 1"]
        y = ["Real 0", "Real 1"]
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
        st.plotly_chart(fig)

        #GRAFICO DE DISPESIÓN
        st.plotly_chart(px.scatter(data, x=var_x, y=var_y, title="Dispersión Y binaria vs X"))

#########################################################################################################

#######MAIN##############################################################################################
def main():
    st.set_page_config(
        page_title="Dashboard",
    )
    with open("estilos.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    with st.sidebar:
        img = Image.open("img/dash.png")
        st.image(img)
        st.header("DASHBOARD")

    menu = ["Inicio", "Análisis univariado", "Análisis de regresión"]
    seleccionar = st.sidebar.selectbox("Menu", menu)

    df, numeric_df, numeric_cols, text_df, text_cols = load_data()

    if seleccionar == "Inicio":
        st.title("DASHBOARD DEL ANÁLISIS DE AIRBNB DE CHICAGO")
        img2 = Image.open("img/inicio.png")
        st.image(img2)
        with st.expander("Descripción", expanded=False):
            st.markdown("El objetivo principal de este dashboard es proporcionar una herramienta accesible e interactiva para explorar relaciones estadísticas entre variables y predecir comportamientos a partir de los datos disponibles.")
        with st.expander("Base de datos de airbnb en chicago", expanded=False):
            df = pd.read_csv("Datos_limpios_chicago.csv")
            st.write(df)
    elif seleccionar == "Análisis univariado":
        univariado(df)
    elif seleccionar == "Análisis de regresión":
        regresion( df, numeric_df, numeric_cols, text_df, text_cols)

if __name__ == '__main__':
    main()

